#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : learn_bert.py
# @Author: sl
# @Date  : 2021/5/20 -  下午10:49

from transformers import pipeline, BertTokenizer, BertForSequenceClassification
from transformers.models.bert import BertModel

from util.nlp_pretrain import NlpPretrain


# padding符号, bert中综合信息符号
PAD, CLS = '[PAD]', '[CLS]'


def test_bert1():
    global tokenizer, model, text
    tokenizer = BertTokenizer.from_pretrained(NlpPretrain.BERT_BASE_UNCASED.path)
    model = BertModel.from_pretrained(NlpPretrain.BERT_BASE_UNCASED.path)
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    print(output)


if __name__ == '__main__':
    # unmasker = pipeline('fill-mask', model='bert-base-uncased')
    # unmasker("Hello I'm a [MASK] model.")

    # test_bert1()

    lin = "中华女子学院：本科层次仅1专业招男生	3"

    content, label = lin.split('\t')

    tokenizer = BertTokenizer.from_pretrained(NlpPretrain.BERT_BASE_UNCASED.path)
    model = BertModel.from_pretrained(NlpPretrain.BERT_BASE_UNCASED.path)

    token = tokenizer.tokenize(content)
    print(token)
    token = [CLS] + token
    seq_len = len(token)
    mask = []
    token_ids = tokenizer.convert_tokens_to_ids(token)

    print(token_ids)

    # BERT
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, do_basic_tokenize=True)
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

    # # OpenAI GPT
    # tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    # model = OpenAIGPTModel.from_pretrained('openai-gpt')
    #
    # # Transformer-XL
    # tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    # model = TransfoXLModel.from_pretrained('transfo-xl-wt103')
    #
    # # OpenAI GPT-2
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # model = GPT2Model.from_pretrained('gpt2')
    #


    pass
