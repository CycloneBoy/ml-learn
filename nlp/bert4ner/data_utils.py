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
    subject: List[object] = None
    guid: str = None

    def __post_init__(self):
        if self.label:
            assert len(self.text) == len(self.label)


def get_entity_bios(seq, id2label):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entity_bio(seq, id2label):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_entities(seq, id2label, markup='bios'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['bio', 'bios']
    if markup == 'bio':
        return get_entity_bio(seq, id2label)
    else:
        return get_entity_bios(seq, id2label)


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
    set_type = get_set_type(input_file)
    lines = read_json(input_file)
    for i, line in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = line['words']
        labels = line['labels']
        subject = get_entities(labels, id2label=None, markup='bios')
        examples.append(Example(text_a, labels, subject, guid))

    return examples


# 读取数据集:json 格式
def read_dataset_txt(input_file):
    """read dataset """
    examples = []
    set_type = get_set_type(input_file)
    with open(input_file, "r", encoding="utf-8") as file:
        text = []
        label = []
        for line in file:
            line = line.strip()
            # 一条文本结束
            if len(line) == 0:
                guid = "%s-%s" % (set_type, len(examples))
                subject = get_entities(label, id2label=None, markup='bios')
                examples.append(Example(text, label, subject, guid))
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

    if label_type == 'bios':
        return bios_tag_list
    else:
        return span_tag_list


def load_tag_from_file(path):
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        tag2id = {tag.strip(): idx for idx, tag in enumerate(lines)}
        id2tag = {idx: tag for tag, idx in tag2id.items()}
    return tag2id, id2tag


def load_tag(path=None, label_type='bios'):
    if path is not None:
        tag2id, id2tag = load_tag_from_file(path)
    else:
        tags = get_labels_from_list(label_type)

        id2tag = {i: label for i, label in enumerate(tags)}
        tag2id = {label: i for i, label in enumerate(tags)}

    return tag2id, id2tag


# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
# tag2id, id2tag = load_tag("./data/tag.txt")
# tag2id, id2tag = load_tag(label_type='bios')
tag2id, id2tag = load_tag(label_type='span')


def get_max_len(data_set):
    max_len = 0
    for index, var in enumerate(data_set):
        if len(var.text) > max_len:
            max_len = len(var.text)
            print(f" {index} - {var}")

    print(f"len: {len(data_set)} - max: {max_len} ")


def get_set_type(input_file):
    """获取数据集类型：train,dev,test """
    begin_index = input_file.rindex("/")
    end_index = input_file.rindex(".")
    set_type = input_file[begin_index + 1:end_index]
    return set_type


# def test_get_id(input_file=WORK_DIR + "/data/train.txt"):
#     set_type = get_set_type(input_file)
#     print(set_type)


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
