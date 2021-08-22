#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict_validate.py
# @Author: sl
# @Date  : 2021/8/22 - 上午9:26

"""
预测检测
"""
from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from nlp.bert4ner.common import load_pickle
from nlp.bert4ner.config import ModelArguments, DataArguments, OurTrainingArguments
from nlp.bert4ner.data_utils import read_data, load_tag
from nlp.bert4ner.train_utils import NERDataset, collate_fn

data_dir = "/home/sl/workspace/python/a2020/ml-learn/nlp/bert4ner/log"


def ner_metrics_test(preds, labels) -> Dict[str, float]:
    """
    该函数是回调函数，Trainer会在进行评估时调用该函数
    """
    preds = preds.flatten()
    labels = labels.flatten()
    assert len(preds) == len(labels)
    # labels为0表示为<pad>，因此计算时需要去掉该部分
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average="macro")
    metrics["precision"] = precision_score(labels, preds, average="macro")
    metrics["recall"] = recall_score(labels, preds, average="macro")
    return metrics


def get_str_len(attention_mask):
    mask_len_list = []
    for row in attention_mask:
        str_len = 0
        for i in row:
            if i > 0:
                str_len += 1
        mask_len_list.append(str_len)

    return mask_len_list


if __name__ == '__main__':
    predict_result = load_pickle("{}/predict_result_{}.pkl".format(data_dir, 420))

    model_args = ModelArguments(use_lstm=False)
    data_args = DataArguments()
    training_args = OurTrainingArguments(train_batch_size=4, eval_batch_size=4)
    # 构建分词器
    tokenizer = BertTokenizer.from_pretrained(training_args.bert_model_name)
    tag2id, id2tag = load_tag(label_type='bios')

    eval_dataset = NERDataset(read_data(data_args.dev_file, data_type="json"), tokenizer=tokenizer)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=training_args.train_batch_size,
                                 collate_fn=collate_fn)

    print(len(predict_result))
    # 11 * 128 * 52
    print(predict_result[0])

    predict_list = []
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    predict_all_mask = np.array([], dtype=int)
    labels_all_mask = np.array([], dtype=int)

    predict_all_mask1 = np.array([], dtype=int)
    labels_all_mask1 = np.array([], dtype=int)

    row_text = []
    for var in predict_result:
        id = var["id"]
        input_ids = var["input_ids"]
        labels = var["labels"]
        predict = var["predict_labels"]
        attention_mask = var["attention_mask"]

        for i, row in enumerate(input_ids):
            text_res = tokenizer.decode(input_ids[i])
            labels_tags = [id2tag[x] for x in labels[i]]
            predict_tags = [id2tag[x] for x in predict[i]]
            mask_res = attention_mask[i]

            res = {}
            res["text"] = text_res
            res["labels1_tags"] = labels_tags
            res["predict_tags"] = predict_tags
            res["mask"] = mask_res
            row_text.append(res)

        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predict)

        mask = attention_mask != 0
        predict_mask = predict[mask]
        labels_mask = labels[mask]

        labels_all_mask = np.append(labels_all_mask, labels_mask)
        predict_all_mask = np.append(predict_all_mask, predict_mask)

        mask_len_list = get_str_len(attention_mask)
        labels_mask1 = []
        for index, row in enumerate(labels):
            label_res = row[1:mask_len_list[index]]
            labels_mask1.append(label_res)

        predict_mask1 = []
        for index, row in enumerate(predict):
            label_res = row[1:mask_len_list[index]]
            predict_mask1.append(label_res)

        labels_all_mask1 = np.append(labels_all_mask1, labels_mask1)
        predict_all_mask1 = np.append(predict_all_mask1, predict_mask1)

    print(len(labels_all))
    print(len(predict_all))

    results = ner_metrics_test(predict_all, labels_all)

    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)

    print()

    print(len(labels_all_mask))
    print(len(predict_all_mask))

    results = ner_metrics_test(predict_all_mask, labels_all_mask)

    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)

    predict_all_mask2 = np.array([], dtype=int)
    labels_all_mask2 = np.array([], dtype=int)

    for row in labels_all_mask1:
        labels_all_mask2 = np.append(labels_all_mask2, row)

    for row in predict_all_mask1:
        predict_all_mask2 = np.append(predict_all_mask2, row)

    print(len(labels_all_mask2))
    print(len(predict_all_mask2))

    results = ner_metrics_test(predict_all_mask2, labels_all_mask2)

    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
