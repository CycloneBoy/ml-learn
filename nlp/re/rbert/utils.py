#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: sl
# @Date  : 2021/8/26 - 上午10:09
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertTokenizer

from nlp.re.rbert.config import DATASET_TYPE_NAME
from util.logger_utils import Logger

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]
E1_BEGIN_INDEX = 0
E1_END_INDEX = 1
E2_BEGIN_INDEX = 2
E2_END_INDEX = 3


def get_labels_from_list(label_type='bios'):
    "CLUENER TAGS"
    bios_tag_list = ["<pad>", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                     'B-organization', 'B-position', 'B-scene', "I-address",
                     "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                     'I-organization', 'I-position', 'I-scene',
                     "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                     'S-name', 'S-organization', 'S-position',
                     'S-scene', 'O', "<start>", "<eos>"]

    span_tag_list = ["O", "address", "book", "company", 'game', 'government', 'movie', 'name', 'organization',
                     'position', 'scene']

    semeval_2010_task8_tag_list = ["Other", "Cause-Effect(e1,e2)", "Cause-Effect(e2,e1)", "Instrument-Agency(e1,e2)",
                                   "Instrument-Agency(e2,e1)", "Product-Producer(e1,e2)", "Product-Producer(e2,e1)",
                                   "Content-Container(e1,e2)", "Content-Container(e2,e1)", "Entity-Origin(e1,e2)",
                                   "Entity-Origin(e2,e1)", "Entity-Destination(e1,e2)", "Entity-Destination(e2,e1)",
                                   "Component-Whole(e1,e2)", "Component-Whole(e2,e1)", "Member-Collection(e1,e2)",
                                   "Member-Collection(e2,e1)", "Message-Topic(e1,e2)", "Message-Topic(e2,e1)"]

    if label_type == 'bios':
        return bios_tag_list
    elif label_type == 'span':
        return span_tag_list
    else:
        return semeval_2010_task8_tag_list


def load_tag_from_file(path):
    result = []
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for idx, line in enumerate(lines):
            result.append(line.strip())
    return result


def load_tag(path=None, label_type='bios'):
    if path is not None:
        tags = load_tag_from_file(path)
    else:
        tags = get_labels_from_list(label_type)

    id2tag = {i: label for i, label in enumerate(tags)}
    tag2id = {label: i for i, label in enumerate(tags)}

    return tags, tag2id, id2tag


def load_tokenizer(model_name_or_path):
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    tags, tag2id, id2tag = load_tag(label_type=DATASET_TYPE_NAME)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, id2tag[pred]))


def now_str(format="%Y-%m-%d_%H"):
    return time.strftime(format, time.localtime())


def init_logger(filename='test'):
    log = Logger('{}_{}.log'.format(filename, now_str()), level='debug')
    return log.logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro"):
    acc = simple_accuracy(preds, labels)
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average=average)
    metrics["precision"] = precision_score(labels, preds, average=average)
    metrics["acc"] = acc
    metrics["recall"] = recall_score(labels, preds, average=average)

    return metrics


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def check_file_exists(filename, delete=False):
    """检查文件是否存在"""
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("文件夹不存在,创建目录:{}".format(dir_name))


def get_checkpoint_dir(file_dir="./"):
    checkpoints = []
    for root, dirs, files in os.walk(file_dir):
        checkpoints.extend(dirs)

    max_step = 0
    checkpoint_dir = None
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else 0
        if int(global_step) > max_step:
            max_step = int(global_step)
            checkpoint_dir = checkpoint
    return checkpoint_dir, max_step


logger = init_logger("./log/test")

if __name__ == '__main__':
    tags, tag2id, id2tag = load_tag(label_type=DATASET_TYPE_NAME)
    logger = init_logger("test")
    logger.info("info")

    checkpoint_dir, max_step = get_checkpoint_dir("/home/sl/workspace/python/a2020/ml-learn/nlp/re/rbert/output")
    print(checkpoint_dir)
