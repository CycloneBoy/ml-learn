#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict_span_validate.py
# @Author: sl
# @Date  : 2021/8/22 - 下午8:18

"""
验证 bert  span
"""
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from nlp.bert4ner.common import load_pickle
from nlp.bert4ner.config import ModelArguments, DataArguments, OurTrainingArguments
from nlp.bert4ner.data_utils import read_data, load_tag, get_entities
from nlp.bert4ner.train_utils import NERDataset, collate_fn
from nlp.bertner.ner_metrics import SpanEntityScore, SeqEntityScore, compute_metric

data_dir = "/home/sl/workspace/python/a2020/ml-learn/nlp/bert4ner/log"


def get_subjects(labels, tag2id, id2tag):
    row_labels = []

    true_labels = []
    for i, row in enumerate(labels):
        res = get_entities(row, id2tag, "bios")
        for sub in res:
            row_labels.append((tag2id[sub[0]], sub[1], sub[2]))

    return row_labels


def show_metric(metric_show):
    print("第二种计算指标")
    eval_info, entity_info = metric_show.result()
    results2 = {f'{key}': value for key, value in eval_info.items()}
    results2['loss'] = 0
    results2["precision"] = results2["acc"]

    print("***** Eval results %s *****" % "prefix")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results2.items()])
    print(info)
    print("***** Entity results %s *****" % "prefix")
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)


if __name__ == '__main__':
    predict_result = load_pickle("{}/predict_result_{}.pkl".format(data_dir, 420))

    model_args = ModelArguments(use_lstm=False)
    data_args = DataArguments()
    training_args = OurTrainingArguments(train_batch_size=4, eval_batch_size=4)
    # 构建分词器
    tokenizer = BertTokenizer.from_pretrained(training_args.bert_model_name)
    tag2id, id2tag = load_tag(label_type='bios')
    span_tag2id, span_id2tag = load_tag(label_type='span')

    eval_dataset = NERDataset(read_data(data_args.dev_file, data_type="json"), tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=training_args.train_batch_size,
                                 collate_fn=collate_fn)

    row_text = []

    metric = SeqEntityScore(id2tag, markup='bios')
    predict_all = []
    labels_all = []

    metric_span = SpanEntityScore(id2tag)

    for var in predict_result:
        id = var["id"]
        input_ids = var["input_ids"]
        labels = var["labels"]
        predict = var["predict_labels"]
        attention_mask = var["attention_mask"]

        true_labels = get_subjects(labels, span_tag2id, id2tag)
        predict_labels = get_subjects(predict, span_tag2id, id2tag)
        metric_span.update(true_labels, predict_labels)

        labels_all.append(labels)
        predict_all.append(predict)
        compute_metric(metric, list(labels), list(predict), tag2id, id2tag)

    show_metric(metric)
    print("-" * 10)
    show_metric(metric_span)

    pass
