#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: sl
# @Date  : 2021/8/27 - 下午4:27
import csv
import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig

from nlp.match.bert.config import BERT_MODEL_NAME, ModelArguments
from nlp.match.bert.model import BertSequenceClassification
from nlp.match.bert.utils import logger, load_tokenizer


@dataclass
class Example:
    guid: str = None
    text_a: str = None
    text_b: str = None
    label: int = None


# 读取数据集:json 格式
def read_dataset_txt(input_file, set_type="train"):
    """read dataset """
    examples = []
    max_length_1 = 0
    max_length_2 = 0
    with open(input_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        for (i, line) in enumerate(reader):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = int(line[2])
            if len(text_a) > max_length_1:
                max_length_1 = len(text_a)
            if len(text_b) > max_length_2:
                max_length_2 = len(text_b)

            if i % 1000 == 0:
                logger.info(line)
            examples.append(Example(guid=guid, text_a=text_a, text_b=text_b, label=label))
    logger.info(f"读取 {set_type} 数据：{len(examples)}, 最大长度：{max_length_1} - {max_length_2} ")
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


class SentenceSimDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=64,
                 tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)):
        self.max_length = 512 if max_length > 512 else max_length
        self.tokenizer = tokenizer
        self.texts = []
        self.input_ids = []

        self.labels = []
        self.masks = []
        self.segments = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_seq_1 = tokenizer.tokenize(example.text_a)
            tokens_seq_2 = tokenizer.tokenize(example.text_b)

            seq, seq_mask, seq_segment, tokens_id = self.truncate_and_pad(tokens_seq_1, tokens_seq_2)

            self.texts.append(seq)
            self.input_ids.append(torch.LongTensor(tokens_id))
            self.masks.append(torch.LongTensor(seq_mask))
            self.segments.append(torch.LongTensor(seq_segment))
            self.labels.append(int(example.label))

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("text: %s" % " ".join([str(x) for x in seq]))
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens_id]))
                logger.info("segment: %s" % " ".join([str(x) for x in seq_segment]))
                logger.info("masks: %s" % " ".join([str(x) for x in seq_mask]))
                logger.info("label: %s" % example.label)

    def truncate_and_pad(self, tokens_seq_1, tokens_seq_2):
        """
        1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
           因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
        2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
        入参:
            seq_1       : 输入序列，在本处其为单个句子。
            seq_2       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，如果是单句，取值都为0；双句按照0/1切分

        """
        # 对超长序列进行截断，sentence1不超过154，sentence2不超过46
        if len(tokens_seq_1) > ((self.max_length - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:((self.max_length - 3) // 2)]
        if len(tokens_seq_2) > ((self.max_length - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:((self.max_length - 3) // 2)]

        # 分别在首尾拼接特殊符号
        seq = [self.tokenizer.cls_token] + tokens_seq_1 + [self.tokenizer.sep_token] + tokens_seq_2 + [
            self.tokenizer.sep_token]
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        tokens_id = self.tokenizer.convert_tokens_to_ids(seq)

        # 创建seq_mask：表明seq长度有意义，padding无意义
        seq_mask = [1] * len(seq)

        assert len(seq) == len(seq_mask)
        assert len(seq) == len(seq_segment)
        assert len(seq) == len(tokens_id)
        return seq, seq_mask, seq_segment, tokens_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "texts": self.texts[item],
            "input_ids": self.input_ids[item],
            "masks": self.masks[item],
            "segments": self.segments[item],
            "labels": self.labels[item],
        }


def collate_fn(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_masks = [feature["masks"] for feature in features]
    batch_segments = [feature["segments"] for feature in features]
    batch_labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)

    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_masks = pad_sequence(batch_masks, batch_first=True, padding_value=0)
    batch_segments = pad_sequence(batch_segments, batch_first=True, padding_value=0)

    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)

    assert batch_input_ids.shape == batch_masks.shape
    assert batch_input_ids.shape == batch_segments.shape
    assert batch_input_ids.shape == batch_attention_mask.shape

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "token_type_ids": batch_segments,
        "labels": batch_labels,
    }


def load_dataset(args, tokenizer, data_type="train"):
    max_length = args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length
    cached_features_file = 'cached_{}-{}_{}_{}_{}'.format(args.model_name, data_type,
                                                          list(filter(None, args.model_name_or_path.split('/'))).pop(),
                                                          str(max_length), args.task_name)

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading dataset from cached file %s", cached_features_file)
        dataset = torch.load(cached_features_file)
        logger.info("Loading dataset success,length: %s", len(dataset))
    else:
        if data_type == "train":
            file_name = args.train_file
        elif data_type == "dev":
            file_name = args.dev_file
        elif data_type == "test":
            file_name = args.test_file
        else:
            file_name = args.dev_file

        logger.info("Creating dataset file at %s", file_name)
        dataset = SentenceSimDataset(read_data(file_name), max_length=max_length, tokenizer=tokenizer)
        torch.save(dataset, cached_features_file)
        logger.info("Catching dataset file at %s,length: %s", cached_features_file, len(dataset))

    return dataset


if __name__ == '__main__':
    # 构建分词器
    tokenizer = load_tokenizer()
    args = ModelArguments(save_steps=100)
    train_dataset = load_dataset(args, tokenizer=tokenizer, data_type="test")
    print(train_dataset[0])

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=64,
                                  collate_fn=collate_fn)

    batch = next(iter(train_dataloader))
    print(batch.keys())
    print(type(batch["input_ids"]))
    print(batch["input_ids"].shape)
    print(type(batch["attention_mask"]))
    print(batch["attention_mask"].shape)
    print(type(batch["token_type_ids"]))
    print(batch["token_type_ids"].shape)
    print(type(batch["labels"]))
    print(batch["labels"].shape)

    args = ModelArguments(num_relations=2)
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name
    )
    model = BertSequenceClassification.from_pretrained(args.model_name_or_path, config=config)

    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
              "token_type_ids": batch["token_type_ids"], "labels": batch["labels"]}

    output = model(**inputs)
    print(type(output))
    print(output[0])
    print(type(output[0]))
    print(output[1])
    print(output[1].shape)
