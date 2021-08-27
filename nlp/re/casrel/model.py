#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: sl
# @Date  : 2021/8/27 - 下午3:22
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class CasRel(BertPreTrainedModel):
    def __init__(self, config, args):
        super(CasRel, self).__init__(config)
        self.config = config
        self.args = args
        self.bert = BertModel(config)
        self.sub_heads_linear = nn.Linear(self.config.hidden_size, 1)
        self.sub_tails_linear = nn.Linear(self.config.hidden_size, 1)
        self.obj_heads_linear = nn.Linear(self.config.hidden_size, self.args.num_relations)
        self.obj_tails_linear = nn.Linear(self.config.hidden_size, self.args.num_relations)

    def get_encoded_text(self, token_ids, attention_mask):
        outputs = self.bert(token_ids, attention_mask=attention_mask)
        #   dim  = (batch_size, seq_len, bert_dim)
        encoded_text = outputs[0]
        return encoded_text

    def get_subs(self, encoded_text):
        #   dim(pred) = (batch_size, seq_len, 1)
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    def get_objs_for_specific_sub(self, sub_head_mapping, sub_tail_mapping, encoded_text):
        # sub_head_mapping [batch, 1, seq] * encoded_text [batch, seq, dim]
        sub_head = torch.matmul(sub_head_mapping, encoded_text)
        sub_tail = torch.matmul(sub_tail_mapping, encoded_text)

        #   dim(sub_head) = dim(sub_tail) = (batch_size, 1, bert_dim)
        sub = (sub_head + sub_tail) / 2
        encoded_text = encoded_text + sub
        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))

        #   dim = (batch_size, seq_len, relation_types)
        return pred_obj_heads, pred_obj_tails

    def forward(self, token_ids, attention_mask, sub_head, sub_tail):
        encoded_text = self.get_encoded_text(token_ids, attention_mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                       encoded_text)

        return {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pre_obj_tails,
        }
