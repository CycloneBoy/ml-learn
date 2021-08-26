#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : albert_learn.py
# @Author: sl
# @Date  : 2021/8/25 - 下午4:55
import torch
from torch.nn.functional import softmax
from transformers import BertTokenizer, AlbertForMaskedLM

if __name__ == '__main__':
    # pretrained = 'voidful/albert_chinese_tiny'
    pretrained = '/home/sl/workspace/data/nlp/voidful/albert_chinese_tiny'
    tokenizer = BertTokenizer.from_pretrained(pretrained)
    model = AlbertForMaskedLM.from_pretrained(pretrained)
    # AlbertModel()

    inputtext = "今天[MASK]情很好"

    maskpos = tokenizer.encode(inputtext, add_special_tokens=True).index(103)

    input_ids = torch.tensor(tokenizer.encode(inputtext, add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels=input_ids)
    loss, prediction_scores = outputs[:2]
    logit_prob = softmax(prediction_scores[0, maskpos], dim=-1).data.tolist()
    predicted_index = torch.argmax(prediction_scores[0, maskpos]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token, logit_prob[predicted_index])
