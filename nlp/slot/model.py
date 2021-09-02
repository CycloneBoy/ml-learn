#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: sl
# @Date  : 2021/9/2 - 下午10:01

import torch
import torch.nn as nn
import torch.nn.functional as F


# 构建slotgate计算方式，利用slot context与intent context
class SlotGate(nn.Module):
    def __init__(self, hidden_dim):
        super(SlotGate, self).__init__()
        self.fc_intent_context = nn.Linear(hidden_dim, hidden_dim)
        self.fc_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, slot_context, intent_context):
        """
        注意这里slot_context是slot上下文context，[batch_size, hidden_dim]，或者是时间步的hidden
        intent_context:[batch_size, hidden_dim]
        """
        # intent_context_linear:[batch_size, hidden_dim]
        intent_context_linear = self.fc_intent_context(intent_context)

        # sum_intent_slot_context:[batch_size, hidden_dim]
        sum_intent_slot_context = slot_context + intent_context_linear

        # fc_linear:[batch_size, hidden_dim]
        fc_linear = self.fc_v(sum_intent_slot_context)

        # sum_gate_vec:[batch_size]
        sum_gate_vec = torch.sum(fc_linear, dim=1)

        return sum_gate_vec


# 这里计算slot context与intent context。就是bigru每个时间步隐藏特征的加权向量，这里不同于原论文，这里使用点乘来计算注意力权重weight
class AttnContext(nn.Module):
    def __init__(self, hidden_dim):
        super(AttnContext, self).__init__()

    def forward(self, hidden, source_output_hidden):
        # source_output_hidden:[batch_size, seq_len, hidden_size]
        # hidden:[batch_size, hidden_size]
        hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]

        attn_weight = torch.sum(hidden * source_output_hidden, dim=2)  # [batch_size, seq_len]

        attn_weight = F.softmax(attn_weight, dim=1).unsqueeze(1)  # [batch_size, 1, seq_len]

        # 类似于注意力向量
        attn_vector = attn_weight.bmm(source_output_hidden)  # [batch_size, 1, hidden_size]

        return attn_vector.squeeze(1)  # [batch_size, hidden_size]


# 构建模型
class BirnnAttentionGate(nn.Module):
    def __init__(self, source_input_dim, source_emb_dim, hidden_dim, n_layers, dropout, pad_index, slot_output_size,
                 intent_output_size, seq_len, predict_flag, slot_attention_flag, device="cpu"):
        super(BirnnAttentionGate, self).__init__()
        self.device = device
        self.pad_index = pad_index
        self.hidden_dim = hidden_dim // 2  # 双向lstm
        self.n_layers = n_layers
        self.slot_output_size = slot_output_size
        # 是否预测模式
        self.predict_flag = predict_flag
        # 原论文中有两种模型结构，一个带slot_attention，一个不带slot_attention
        self.slot_attention_flag = slot_attention_flag

        self.source_embedding = nn.Embedding(source_input_dim, source_emb_dim, padding_idx=pad_index)
        # 双向gru，隐层维度是hidden_dim
        self.source_gru = nn.GRU(source_emb_dim, self.hidden_dim, n_layers, dropout=dropout, bidirectional=True,
                                 batch_first=True)  # 使用双向

        # slot context
        self.slot_context = AttnContext(hidden_dim)

        # intent context
        self.intent_context = AttnContext(hidden_dim)

        # slotgate类
        self.slotGate = SlotGate(hidden_dim)

        # 意图intent预测
        self.intent_output = nn.Linear(hidden_dim, intent_output_size)

        # 槽slot预测
        self.slot_output = nn.Linear(hidden_dim, slot_output_size)

    def forward(self, source_input, source_len):
        '''
        source_input:[batch_size, seq_len]
        source_len:[batch_size]
        '''
        if self.predict_flag:
            assert len(source_input) == 1, '预测时一次输入一句话'
            seq_len = source_len[0]

            # 将输入的source进行编码
            # source_embedded:[batch_size, seq_len, source_emb_dim]
            source_embedded = self.source_embedding(source_input)
            packed = torch.nn.utils.rnn.pack_padded_sequence(source_embedded, source_len, batch_first=True,
                                                             enforce_sorted=True)  # 这里enfore_sotred=True要求数据根据词数排序
            source_output, hidden = self.source_gru(packed)
            # source_output=[batch_size, seq_len, 2 * self.hidden_size]，这里的2*self.hidden_size = hidden_dim
            # hidden=[n_layers * 2, batch_size, self.hidden_size]
            source_output, _ = torch.nn.utils.rnn.pad_packed_sequence(source_output, batch_first=True,
                                                                      padding_value=self.pad_index, total_length=len(
                    source_input[0]))  # 这个会返回output以及压缩后的legnths

            batch_size = source_input.shape[0]
            seq_len = source_input.shape[1]
            # 保存slot的预测概率
            slot_outputs = torch.zeros(batch_size, seq_len, self.slot_output_size).to(self.device)

            aligns = source_output.transpose(0, 1)  # 为了拿到每个时间步的输出特征，即每个时间步的隐藏向量

            output_tokens = []

            # 槽识别
            for t in range(seq_len):
                '''
                此时刻时间步的输出隐向量
                '''
                aligned = aligns[t]  # [batch_size, hidden_size]

                # 是否需要计算slot attention
                if self.slot_attention_flag:

                    # [batch_size, hidden_size]
                    slot_context = self.slot_context(aligned, source_output)

                    # [batch_size, hidden_size]，意图上下文向量，利用bigru最后一个时间步的隐状态
                    intent_context = self.intent_context(source_output[:, -1, :], source_output)

                    # gate机制，[batch_size]
                    slot_gate = self.slotGate(slot_context, intent_context)

                    # slot_gate:[batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)

                    # slot_context_gate:[batch_size, hidden_dim]
                    slot_context_gate = slot_gate * slot_context

                # 否则，利用每个时间步的隐状态与intent context计算slot gate
                else:
                    # [batch_size, hidden_size]，意图上下文向量，利用bigru最后一个时间步的隐状态
                    intent_context = self.intent_context(source_output[:, -1, :], source_output)

                    # gate机制，[batch_size]
                    slot_gate = self.slotGate(source_output[:, t, :], intent_context)

                    # slot_gate:[batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)

                    # slot_context_gate:[batch_size, hidden_dim]
                    slot_context_gate = slot_gate * source_output[:, t, :]

                # 预测槽slot, [batch_size, slot_output_size]
                slot_prediction = self.slot_output(slot_context_gate + source_output[:, t, :])
                slot_outputs[:, t, :] = slot_prediction

            # 意图识别
            intent_outputs = self.intent_output(intent_context + source_output[:, -1, :])

            return slot_outputs, intent_outputs

        # 训练阶段
        else:
            # 将输入的source进行编码
            # source_embedded:[batch_size, seq_len, source_emb_dim]
            source_embedded = self.source_embedding(source_input)
            packed = torch.nn.utils.rnn.pack_padded_sequence(source_embedded, source_len, batch_first=True,
                                                             enforce_sorted=True)  # 这里enfore_sotred=True要求数据根据词数排序
            source_output, hidden = self.source_gru(packed)
            # source_output=[batch_size, seq_len, 2 * self.hidden_size]，这里的2*self.hidden_size = hidden_dim
            # hidden=[n_layers * 2, batch_size, self.hidden_size]
            source_output, _ = torch.nn.utils.rnn.pad_packed_sequence(source_output, batch_first=True,
                                                                      padding_value=self.pad_index, total_length=len(
                    source_input[0]))  # 这个会返回output以及压缩后的legnths

            batch_size = source_input.shape[0]
            seq_len = source_input.shape[1]
            # 保存slot的预测概率
            slot_outputs = torch.zeros(batch_size, seq_len, self.slot_output_size).to(self.device)

            aligns = source_output.transpose(0, 1)  # 为了拿到每个时间步的输出特征，即每个时间步的隐藏向量

            # 槽识别
            for t in range(seq_len):
                '''
                此时刻时间步的输出隐向量
                '''
                aligned = aligns[t]  # [batch_size, hidden_size]

                # 是否需要计算slot attention
                if self.slot_attention_flag:

                    # [batch_size, hidden_size]
                    slot_context = self.slot_context(aligned, source_output)

                    # [batch_size, hidden_size]，意图上下文向量，利用bigru最后一个时间步的隐状态
                    intent_context = self.intent_context(source_output[:, -1, :], source_output)

                    # gate机制，[batch_size]
                    slot_gate = self.slotGate(slot_context, intent_context)

                    # slot_gate:[batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)

                    # slot_context_gate:[batch_size, hidden_dim]
                    slot_context_gate = slot_gate * slot_context

                # 否则，利用每个时间步的隐状态与intent context计算slot gate
                else:
                    # [batch_size, hidden_size]，意图上下文向量，利用bigru最后一个时间步的隐状态
                    intent_context = self.intent_context(source_output[:, -1, :], source_output)

                    # gate机制，[batch_size]
                    slot_gate = self.slotGate(source_output[:, t, :], intent_context)

                    # slot_gate:[batch_size, 1]
                    slot_gate = slot_gate.unsqueeze(1)

                    # slot_context_gate:[batch_size, hidden_dim]
                    slot_context_gate = slot_gate * source_output[:, t, :]

                # 预测槽slot, [batch_size, slot_output_size]
                slot_prediction = self.slot_output(slot_context_gate + source_output[:, t, :])
                slot_outputs[:, t, :] = slot_prediction

            # 意图识别
            intent_outputs = self.intent_output(intent_context + source_output[:, -1, :])

            return slot_outputs, intent_outputs
