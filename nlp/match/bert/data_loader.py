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
from nlp.match.bert.utils import load_tokenizer, load_vocab, get_char_list, get_word_list, UNK_TOKEN, PAD_TOKEN, \
    pad_seq
from util.logger_utils import logger


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


class SentenceSimDatasetForRnn(Dataset):
    def __init__(self, examples: List[Example], vocab_file, max_length=64, mode="char"):
        # char embedding or word embedding
        self.mode = mode
        self.text_a_list = []
        self.text_a_token = []
        self.text_a_length = []
        self.text_a_mask = []
        self.text_b_list = []
        self.text_b_token = []
        self.text_b_length = []
        self.text_b_mask = []

        self.labels = []
        # load vocab
        word2idx, idx2word, vocab = load_vocab(vocab_file)

        self.max_length = max_length

        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            if mode == "char":
                text_a = get_char_list(example.text_a)
                text_b = get_char_list(example.text_b)
            else:
                text_a = get_word_list(example.text_a)
                text_b = get_word_list(example.text_b)

            self.text_a_list.append(text_a)
            self.text_b_list.append(text_b)
            self.labels.append(int(example.label))
            # 1为[UNK]对应的索引
            text_a_token = [word2idx[word] if word in word2idx.keys() else word2idx[UNK_TOKEN] for word in text_a]
            text_b_token = [word2idx[word] if word in word2idx.keys() else word2idx[UNK_TOKEN] for word in text_b]

            text_a_length = min(len(text_a_token), max_length)
            text_b_length = min(len(text_b_token), max_length)

            self.text_a_length.append(text_a_length)
            self.text_b_length.append(text_b_length)

            pad_text_a_token = pad_seq(text_a_token, max_len=max_length, value=word2idx[PAD_TOKEN])
            pad_text_b_token = pad_seq(text_b_token, max_len=max_length, value=word2idx[PAD_TOKEN])

            self.text_a_token.append(torch.LongTensor(pad_text_a_token))
            self.text_b_token.append(torch.LongTensor(pad_text_b_token))

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("text_a: %s" % " ".join([str(x) for x in example.text_a]))
                logger.info("text_a_token: %s" % " ".join([str(x) for x in text_a_token]))
                logger.info("text_a_length: %s" % text_a_length)
                logger.info("text_b: %s" % " ".join([str(x) for x in example.text_b]))
                logger.info("text_b_token: %s" % " ".join([str(x) for x in text_b_token]))
                logger.info("text_b_length: %s" % text_b_length)
                logger.info("label: %s" % example.label)

        logger.info(f"读取完毕:{len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {
            "text_a_token": self.text_a_token[item],
            "text_a_length": self.text_a_length[item],
            "text_b_token": self.text_b_token[item],
            "text_b_length": self.text_b_length[item],
            "labels": self.labels[item],
        }


def collate_fn_rnn(features) -> Dict[str, Tensor]:
    batch_text_a_token = [feature["text_a_token"] for feature in features]
    batch_text_a_length = torch.tensor([feature["text_a_length"] for feature in features], dtype=torch.long)
    batch_text_b_token = [feature["text_b_token"] for feature in features]
    batch_text_b_length = torch.tensor([feature["text_b_length"] for feature in features], dtype=torch.long)
    batch_labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)

    # padding
    batch_text_a_token = pad_sequence(batch_text_a_token, batch_first=True, padding_value=0)
    batch_text_b_token = pad_sequence(batch_text_b_token, batch_first=True, padding_value=0)

    return {
        "text_a_token": batch_text_a_token,
        "text_a_length": batch_text_a_length,
        "text_b_token": batch_text_b_token,
        "text_b_length": batch_text_b_length,
        "labels": batch_labels,
    }


def load_dataset(args, tokenizer=None, data_type="train"):
    max_length = args.train_max_seq_length if data_type == 'train' else args.eval_max_seq_length
    if str(args.model_name).lower().find("bert") > -1:
        cached_features_file = 'cached_{}-{}_{}_{}_{}'.format(args.model_name, data_type,
                                                              list(filter(None,
                                                                          args.model_name_or_path.split('/'))).pop(),
                                                              str(max_length), args.task_name)
    else:
        cached_features_file = 'cached_{}-{}_{}_{}'.format(args.model_name, data_type, str(max_length), args.task_name)

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
        if str(args.model_name).lower().find("bert") > -1:
            dataset = SentenceSimDataset(read_data(file_name), max_length=max_length, tokenizer=tokenizer)
        else:
            dataset = SentenceSimDatasetForRnn(read_data(file_name), vocab_file=args.vocab_file, max_length=max_length,
                                               mode=args.vocab_mode)
        torch.save(dataset, cached_features_file)
        logger.info("Catching dataset file at %s,length: %s", cached_features_file, len(dataset))

    return dataset


if __name__ == '__main__':
    logger.info('tes')

    args = ModelArguments(save_steps=100, num_relations=2)
    # 构建分词器
    tokenizer = load_tokenizer(args=args)
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

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name
    )
    model = BertSequenceClassification.from_pretrained(args.model_name_or_path, config=config, args=args)

    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
              "token_type_ids": batch["token_type_ids"], "labels": batch["labels"]}

    output = model(**inputs)
    print(type(output))
    print(output[0])
    print(type(output[0]))
    print(output[1])
    print(output[1].shape)
