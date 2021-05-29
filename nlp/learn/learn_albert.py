#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : learn_albert.py
# @Author: sl
# @Date  : 2021/5/28 -  下午10:02


from transformers import *
import torch
from torch.nn.functional import softmax
from transformers.modeling_outputs import MaskedLMOutput

from util.nlp_pretrain import NlpPretrain

if __name__ == '__main__':
    pretrained = NlpPretrain.ALBERT_CHINESE_TINY.path
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = AlbertForMaskedLM.from_pretrained(pretrained)

    inputtext = "今天[MASK]情很好"
    inputtext = "今天空气清新,阳光明[MASK]"

    maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)
    print(maskpos)

    input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, attention_mask=input_ids, return_dict=True)

    # loss, prediction_scores = outputs
    # print(outputs)
    prediction_scores = outputs.logits

    logit_prob = softmax(prediction_scores[0, maskpos]).data.tolist()

    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, logit_prob[predicted_index])
