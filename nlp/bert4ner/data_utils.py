#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_utils.py
# @Author: sl
# @Date  : 2021/8/20 - 上午8:40
import json
from dataclasses import dataclass
from typing import List

from transformers import BertTokenizer

from nlp.bert4ner.config import BERT_PATH, WORK_DIR, CLUENER_DATASET_DIR


@dataclass
class Example:
    text: List[str]
    label: List[str] = None

    def __post_init__(self):
        if self.label:
            assert len(self.text) == len(self.label)


# 读取数据集:json 格式
def read_json(input_file):
    """read dataset """
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label', None)
            words = list(text)
            labels = ['O'] * len(words)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index + 1]) == sub_name
                            if start_index == end_index:
                                labels[start_index] = 'S-' + key
                            else:
                                labels[start_index] = 'B-' + key
                                labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            lines.append({"words": words, "labels": labels})
    return lines


def read_dataset_json(input_file):
    """ 读取数据集:json 格式  """
    examples = []
    lines = read_json(input_file)
    for line in lines:
        examples.append(Example(line["words"], line["labels"]))

    return examples


# 读取数据集:json 格式
def read_dataset_txt(input_file):
    """read dataset """
    examples = []
    with open(input_file, "r", encoding="utf-8") as file:
        text = []
        label = []
        for line in file:
            line = line.strip()
            # 一条文本结束
            if len(line) == 0:
                examples.append(Example(text, label))
                text = []
                label = []
                continue
            text.append(line.split()[0])
            label.append(line.split()[1])
    return examples


def read_data(path, data_type="txt"):
    examples = None
    if data_type == 'txt':
        examples = read_dataset_txt(path)
    elif data_type == "json":
        examples = read_dataset_json(path)

    return examples


def get_labels_from_list():
    "CLUENER TAGS"
    return ["<pad>", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
            'B-organization', 'B-position', 'B-scene', "I-address",
            "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
            'I-organization', 'I-position', 'I-scene',
            "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
            'S-name', 'S-organization', 'S-position',
            'S-scene', 'O', "<start>", "<eos>"]


def load_tag_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        tag2id = {tag.strip(): idx for idx, tag in enumerate(lines)}
        id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag


def load_tag(path=None):
    if path is not None:
        tag2id, id2tag = load_tag_from_file(path)
    else:
        tags = get_labels_from_list()

        id2tag = {i: label for i, label in enumerate(tags)}
        tag2id = {label: i for i, label in enumerate(tags)}

    return tag2id, id2tag


# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
tag2id, id2tag = load_tag("./data/tag.txt")


def get_max_len(data_set):
    max_len = 0
    for index, var in enumerate(data_set):
        if len(var.text) > max_len:
            max_len = len(var.text)
            print(f" {index} - {var}")

    print(f"len: {len(data_set)} - max: {max_len} ")


if __name__ == "__main__":

    train_data = read_data(WORK_DIR + "/data/train.txt")
    eval_data = read_data(WORK_DIR + "/data/dev.txt")

    print("-> data")
    print(train_data[0])

    get_max_len(train_data)
    get_max_len(eval_data)

    for i in range(10):
        print(train_data[i])

    print("-> tag")
    tag2id, id2tag = load_tag(WORK_DIR + "/data/tag.txt")
    print(tag2id)
    print(id2tag)

    print()
    print("-> CLUENER")
    train_data = read_data(CLUENER_DATASET_DIR + "/train.json", "json")
    get_max_len(train_data)

    print("-> tag")
    tag2id, id2tag = load_tag(CLUENER_DATASET_DIR + "/tag.txt")
    print(tag2id)
    print(id2tag)

    pass
    """
    examples = read_data("./data/train.txt")
    print(examples[0])
    tag2id, id2tag = load_tag("./data/tag.txt")
    print(tag2id)
    print(id2tag)
    """
