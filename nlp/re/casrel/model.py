#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: sl
# @Date  : 2021/8/27 - 下午3:22
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return encoded_text, outputs

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

    # define the loss function
    def loss_fn(self, gold, pred, mask):
        pred = pred.squeeze(-1)
        loss = F.binary_cross_entropy(pred, gold, reduction='none')
        if loss.shape != mask.shape:
            mask = mask.unsqueeze(-1)
        loss = torch.sum(loss * mask) / torch.sum(mask)
        return loss

    def forward(self, input_ids=None, attention_mask=None, sub_head=None, sub_tail=None,
                sub_heads=None, sub_tails=None, obj_heads=None, obj_tails=None,
                output_attentions=None, output_hidden_states=None, return_dict=False, ):
        encoded_text, outputs = self.get_encoded_text(input_ids, attention_mask)
        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)
        sub_head_mapping = sub_head.unsqueeze(1)
        sub_tail_mapping = sub_tail.unsqueeze(1)
        pred_obj_heads, pred_obj_tails = self.get_objs_for_specific_sub(sub_head_mapping, sub_tail_mapping,
                                                                        encoded_text)

        logits = {
            "sub_heads": pred_sub_heads,
            "sub_tails": pred_sub_tails,
            "obj_heads": pred_obj_heads,
            "obj_tails": pred_obj_tails,
        }
        outputs = (logits,) + outputs[2:]

        loss = None
        if sub_heads is not None and sub_tails is not None and obj_heads is not None and obj_tails is not None:
            sub_heads_loss = self.loss_fn(sub_heads, pred_sub_heads, attention_mask)
            sub_tails_loss = self.loss_fn(sub_tails, pred_sub_tails, attention_mask)
            obj_heads_loss = self.loss_fn(obj_heads, pred_obj_heads, attention_mask)
            obj_tails_loss = self.loss_fn(obj_tails, pred_obj_tails, attention_mask)
            loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)
            outputs = (loss,) + outputs

        return outputs
