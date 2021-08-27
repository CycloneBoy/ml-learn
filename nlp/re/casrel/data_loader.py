#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: sl
# @Date  : 2021/8/27 - 下午4:27
import json
from collections import defaultdict
from dataclasses import dataclass
from random import choice
from typing import List, Dict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import BertTokenizer

from nlp.re.casrel.config import BERT_MODEL_NAME
from nlp.re.casrel.utils import load_tag
from nlp.re.casrel.utils import logger


@dataclass
class Example:
    guid: str = None
    text: str = None
    spo_list: List = None


@dataclass
class ReEntity:
    predicate: str = None
    object_type: str = None
    subject_type: str = None
    object: str = None
    subject: str = None


# 读取数据集:json 格式
def read_dataset_txt(input_file, set_type="train"):
    """read dataset """
    examples = []
    with open(input_file, "r", encoding="utf-8") as file:
        data = file.readlines()
    res = [json.loads(i) for i in data]
    for (i, line) in enumerate(res):
        guid = "%s-%s" % (set_type, i)
        text = line["text"]
        spo_list = line["spo_list"] if "spo_list" in line else []
        re_entitys = []
        if len(spo_list) > 0:
            for spo in spo_list:
                re_entitys.append(ReEntity(spo["predicate"], spo["object_type"],
                                           spo["subject_type"], spo["object"], spo["subject"]))
        if i % 1000 == 0:
            logger.info(line)
        examples.append(Example(guid=guid, text=text, spo_list=re_entitys))
    return examples


def read_data(path, set_type="train"):
    examples = read_dataset_txt(path, set_type)
    return examples


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


class Re18BaiduDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=384,
                 tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)):
        self.max_length = 512 if max_length > 512 else max_length
        self.tags, self.tag2id, self.id2tag = load_tag()
        self.tokenizer = tokenizer
        self.texts = []
        self.input_ids = []

        self.sub_heads = []
        self.sub_tails = []
        self.sub_head = []
        self.sub_tail = []
        self.obj_heads = []
        self.obj_tails = []
        self.spo_list = []

        self.token_type_ids = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = tokenizer.tokenize(example.text)
            if len(tokens) > max_length - 1:
                tokens = tokens[: (max_length - 1)]

            token_type_ids = [0] * len(tokens)

            # add cls　token
            tokens = [tokenizer.cls_token] + tokens
            token_type_ids = [0] + token_type_ids
            text_len = len(tokens)

            sub_heads, sub_tails = torch.zeros(text_len), torch.zeros(text_len)
            sub_head, sub_tail = torch.zeros(text_len), torch.zeros(text_len)
            obj_heads = torch.zeros((text_len, len(self.tags)))
            obj_tails = torch.zeros((text_len, len(self.tags)))

            s2ro_map = defaultdict(list)
            for spo in example.spo_list:
                triple = (self.tokenizer(spo.subject, add_special_tokens=False)['input_ids'],
                          self.tag2id(spo.predicate),
                          self.tokenizer(spo.object, add_special_tokens=False)['input_ids'])
                sub_head_idx = find_head_idx(tokens, triple[0])
                obj_head_idx = find_head_idx(tokens, triple[2])
                if sub_head_idx != -1 and obj_head_idx != -1:
                    sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                    s2ro_map[sub].append(
                        (obj_head_idx, obj_head_idx + len(triple[2]) - 1, triple[1]))

            if s2ro_map:
                for s in s2ro_map:
                    sub_heads[s[0]] = 1
                    sub_tails[s[1]] = 1
                sub_head_idx, sub_tail_idx = choice(list(s2ro_map.keys()))
                sub_head[sub_head_idx] = 1
                sub_tail[sub_tail_idx] = 1
                for ro in s2ro_map.get((sub_head_idx, sub_tail_idx), []):
                    obj_heads[ro[0]][ro[2]] = 1
                    obj_tails[ro[1]][ro[2]] = 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            self.texts.append(tokens)
            self.input_ids.append(torch.LongTensor(input_ids))
            self.sub_heads.append(torch.LongTensor(sub_heads))
            self.sub_tails.append(torch.LongTensor(sub_tails))
            self.sub_head.append(torch.LongTensor(sub_head))
            self.sub_tail.append(torch.LongTensor(sub_tail))

            self.obj_heads.append(torch.LongTensor(obj_heads))
            self.obj_tails.append(torch.LongTensor(obj_tails))

            self.spo_list.append(example.spo_list)

            self.token_type_ids.append(torch.LongTensor(token_type_ids))

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("sub_heads: %s" % " ".join([str(x) for x in sub_heads]))
                logger.info("sub_tails: %s" % " ".join([str(x) for x in sub_tails]))

                logger.info("sub_head: %s" % " ".join([str(x) for x in sub_head]))
                logger.info("sub_tail: %s" % " ".join([str(x) for x in sub_tail]))
                logger.info("obj_heads: %s" % " ".join([str(x) for x in obj_heads]))
                logger.info("obj_tails: %s" % " ".join([str(x) for x in obj_tails]))
                logger.info("spo_list: %s" % " ".join([str(x) for x in example.spo_list]))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "texts": self.texts[item],
            "input_ids": self.input_ids[item],
            "token_type_ids": self.token_type_ids[item],
            "sub_heads": self.sub_heads[item],
            "sub_tails": self.sub_tails[item],
            "sub_head": self.sub_head[item],
            "sub_tail": self.sub_tail[item],
            "obj_heads": self.obj_heads[item],
            "obj_tails": self.obj_tails[item],
            "spo_list": self.spo_list[item],
        }


def collate_fn(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_token_type_ids = [feature["token_type_ids"] for feature in features]
    batch_sub_heads = [feature["sub_heads"] for feature in features]
    batch_sub_tails = [feature["sub_tails"] for feature in features]
    batch_sub_head = [feature["sub_head"] for feature in features]
    batch_sub_tail = [feature["sub_tail"] for feature in features]
    batch_obj_heads = [feature["obj_heads"] for feature in features]
    batch_obj_tails = [feature["obj_tails"] for feature in features]

    batch_spo_list = [feature["spo_list"] for feature in features]

    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True, padding_value=0)
    batch_sub_heads = pad_sequence(batch_sub_heads, batch_first=True, padding_value=0)
    batch_sub_tails = pad_sequence(batch_sub_tails, batch_first=True, padding_value=0)
    batch_sub_head = pad_sequence(batch_sub_head, batch_first=True, padding_value=0)
    batch_sub_tail = pad_sequence(batch_sub_tail, batch_first=True, padding_value=0)

    batch_obj_heads = pad_sequence(batch_obj_heads, batch_first=True, padding_value=0)
    batch_obj_tails = pad_sequence(batch_obj_tails, batch_first=True, padding_value=0)

    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

    assert batch_input_ids.shape == batch_token_type_ids.shape
    assert batch_input_ids.shape == batch_sub_heads.shape
    assert batch_input_ids.shape == batch_sub_tails.shape

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "token_type_ids": batch_token_type_ids,
        "sub_head": batch_sub_head,
        "sub_tail": batch_sub_tail,
        "sub_heads": batch_sub_heads,
        "sub_tails": batch_sub_tails,
        "obj_heads": batch_obj_heads,
        "obj_tails": batch_obj_tails,
        "triples": batch_spo_list
    }


if __name__ == '__main__':
    pass
