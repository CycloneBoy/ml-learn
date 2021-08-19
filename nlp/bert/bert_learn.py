#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2021/8/4 10:49
# @Author  : shenglei
# @Site    : 
# @File    : bert_learn.py

from transformers import BertTokenizer, BertModel, BertForTokenClassification

if __name__ == '__main__':
    # unmasker = pipeline('fill-mask', model='bert-base-uncased')
    #
    # res = unmasker("Hello I'm a [MASK] model.")
    #
    # print(res)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model = BertModel.from_pretrained("bert-base-uncased",mirror='tuna')
    model = BertModel.from_pretrained("bert-base-chinese", mirror='tuna')
    text = "Replace me by any text you'd like."
    text = "GPU并行的其他注意事项小可爱们可以移步这里：Pytorch | 多GPU并行 DataParallel"
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(output)

    BertForTokenClassification
