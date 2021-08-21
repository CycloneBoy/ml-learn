#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: sl
# @Date  : 2021/8/20 - 上午8:43

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import TokenClassifierOutput, QuestionAnsweringModelOutput

from nlp.bert4ner.layers.crf import CRF
from nlp.bert4ner.layers.linears import PoolerStartLogits, PoolerEndLogits


class BertForNER(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        if "model_args" in model_kargs:
            model_args = model_kargs["model_args"]
            # 必须将额外的参数更新至self.config中，不然再使用from_pretrained方法加载训练好的模型时会出错
            self.config.__dict__.update(model_args.__dict__)
        self.num_labels = self.config.ner_num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.lstm = nn.LSTM(self.config.hidden_size,
                            self.config.lstm_hidden_size,
                            num_layers=self.config.lstm_layers,
                            dropout=self.config.lstm_dropout,
                            bidirectional=True,
                            batch_first=True)
        if self.config.use_lstm:
            self.classifier = nn.Linear(self.config.lstm_hidden_size * 2, self.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(outputs[0])
        if self.config.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BertCrfForNER(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        if "model_args" in model_kargs:
            model_args = model_kargs["model_args"]
            # 必须将额外的参数更新至self.config中，不然再使用from_pretrained方法加载训练好的模型时会出错
            self.config.__dict__.update(model_args.__dict__)
        self.num_labels = self.config.ner_num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout)
        self.lstm = nn.LSTM(self.config.hidden_size,
                            self.config.lstm_hidden_size,
                            num_layers=self.config.lstm_layers,
                            dropout=self.config.lstm_dropout,
                            bidirectional=True,
                            batch_first=True)
        if self.config.use_lstm:
            self.classifier = nn.Linear(self.config.lstm_hidden_size * 2, self.num_labels)
        else:
            self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)
        self.crf = CRF(num_tags=self.num_labels, batch_first=True)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        bert_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = self.dropout(bert_outputs[0])
        if self.config.use_lstm:
            sequence_output, _ = self.lstm(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs

        if not return_dict:
            return outputs  # (loss), scores

        return TokenClassifierOutput(
            loss=-1 * loss if loss is not None else None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class BertSpanForNER(BertPreTrainedModel):
    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        if "model_args" in model_kargs:
            model_args = model_kargs["model_args"]
            # 必须将额外的参数更新至self.config中，不然再使用from_pretrained方法加载训练好的模型时会出错
            self.config.__dict__.update(model_args.__dict__)
        self.num_labels = self.config.ner_num_labels
        self.soft_label = self.config.soft_label
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(self.config.hidden_dropout)

        self.start_fc = PoolerStartLogits(self.config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(self.config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(self.config.hidden_size + 1, self.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            pos=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            start_positions=None,
            end_positions=None,
            input_len=None,
            segment_ids=None,
            subjects_id=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = self.dropout(outputs[0])

        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        loss = None
        if start_positions is not None and end_positions is not None and attention_mask is not None:
            loss_fct = nn.CrossEntropyLoss()

            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)

            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1

            active_start_logits = start_logits.view(-1, self.num_labels)
            active_end_logits = end_logits.view(-1, self.num_labels)

            active_start_labels = torch.where(
                active_loss, start_positions.view(-1), torch.tensor(loss_fct.ignore_index).type_as(start_positions)
            )
            active_end_labels = torch.where(
                active_loss, end_positions.view(-1), torch.tensor(loss_fct.ignore_index).type_as(end_positions)
            )

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            loss = (start_loss + end_loss) / 2
            outputs = (loss,) + outputs

        if not return_dict:
            return outputs

        return QuestionAnsweringModelOutput(
            loss=loss if loss is not None else None,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=None,
            attentions=None,
        )
