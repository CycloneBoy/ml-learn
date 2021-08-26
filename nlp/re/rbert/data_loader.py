#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_loader.py
# @Author: sl
# @Date  : 2021/8/26 - 上午11:18
import csv
import time
from dataclasses import dataclass
from typing import List, Dict

import torch
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertConfig

from nlp.re.rbert.config import BERT_MODEL_NAME, DATASET_TYPE_NAME, BERT_PATH, ModelArguments
from nlp.re.rbert.model import RBERT
from nlp.re.rbert.utils import load_tag, load_tokenizer, logger


# logger = logging.getLogger(__name__)


@dataclass
class Example:
    guid: str = None
    text: str = None
    label: str = None


# 读取数据集:json 格式
def read_dataset_txt(input_file, set_type="train"):
    """read dataset """
    examples = []
    tags, tag2id, id2tag = load_tag(label_type=DATASET_TYPE_NAME)
    with open(input_file, "r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t", quotechar=None)
        for (i, line) in enumerate(reader):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = tag2id[line[0]]
            if i % 1000 == 0:
                logger.info(line)
            examples.append(Example(guid=guid, text=text_a, label=label))
    return examples


def read_data(path, set_type="train"):
    examples = read_dataset_txt(path, set_type)
    return examples


class SemevalTask8Dataset(Dataset):
    def __init__(self, examples: List[Example], max_length=384,
                 tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)):
        self.max_length = 512 if max_length > 512 else max_length
        self.tags, self.tag2id, self.id2tag = load_tag(label_type=DATASET_TYPE_NAME)
        self.texts = []
        self.labels = []
        self.e1_masks = []
        self.e2_masks = []
        self.token_type_ids = []
        self.input_ids = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens_a = tokenizer.tokenize(example.text)
            if len(tokens_a) > max_length - 1:
                tokens_a = tokens_a[: (max_length - 1)]

            e11_p = tokens_a.index("<e1>")  # the start position of entity1
            e12_p = tokens_a.index("</e1>")  # the end position of entity1
            e21_p = tokens_a.index("<e2>")  # the start position of entity2
            e22_p = tokens_a.index("</e2>")  # the end position of entity2

            # Replace the token
            tokens_a[e11_p] = "$"
            tokens_a[e12_p] = "$"
            tokens_a[e21_p] = "#"
            tokens_a[e22_p] = "#"

            # Add 1 because of the [CLS] token
            e11_p += 1
            e12_p += 1
            e21_p += 1
            e22_p += 1

            tokens = tokens_a
            token_type_ids = [0] * len(tokens)

            # add cls　token
            tokens = [tokenizer.cls_token] + tokens
            token_type_ids = [0] + token_type_ids

            # e1 mask, e2 mask
            e1_mask = [0] * len(tokens)
            e2_mask = [0] * len(tokens)

            for i in range(e11_p, e12_p + 1):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p + 1):
                e2_mask[i] = 1

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            label_id = int(example.label)
            self.texts.append(tokens)
            self.labels.append(label_id)
            self.e1_masks.append(torch.LongTensor(e1_mask))
            self.e2_masks.append(torch.LongTensor(e2_mask))
            self.token_type_ids.append(torch.LongTensor(token_type_ids))
            self.input_ids.append(torch.LongTensor(input_ids))

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("label: %s (id = %d)" % (self.id2tag[example.label], label_id))
                logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        for text, mask in zip(self.texts, self.e1_masks):
            assert len(text) == len(mask)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "input_ids": self.input_ids[item],
            "labels": self.labels[item],
            "token_type_ids": self.token_type_ids[item],
            "e1_mask": self.e1_masks[item],
            "e2_mask": self.e2_masks[item],
            "token": self.texts[item],
        }


def collate_fn_rbert(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_token_type_ids = [feature["token_type_ids"] for feature in features]
    batch_e1_mask = [feature["e1_mask"] for feature in features]
    batch_e2_mask = [feature["e2_mask"] for feature in features]

    batch_labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)

    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
    batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True, padding_value=0)
    batch_e1_mask = pad_sequence(batch_e1_mask, batch_first=True, padding_value=0)
    batch_e2_mask = pad_sequence(batch_e2_mask, batch_first=True, padding_value=0)
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    # batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=0)

    assert batch_input_ids.shape == batch_token_type_ids.shape
    assert batch_input_ids.shape == batch_e1_mask.shape
    assert batch_input_ids.shape == batch_e2_mask.shape
    assert batch_input_ids.shape == batch_attention_mask.shape

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "token_type_ids": batch_token_type_ids,
        "labels": batch_labels,
        "e1_mask": batch_e1_mask,
        "e2_mask": batch_e2_mask
    }


if __name__ == '__main__':
    # 构建分词器
    tokenizer = load_tokenizer(BERT_PATH)

    train_filename = "/home/sl/workspace/python/a2020/ml-learn/data/nlp/SemevalTask8/train.tsv"

    # 构建dataset
    train_dataset = SemevalTask8Dataset(read_data(train_filename), tokenizer=tokenizer)
    print(train_dataset[0])

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=32,
                                  collate_fn=collate_fn_rbert)

    batch = next(iter(train_dataloader))
    print(batch.keys())
    print(type(batch["input_ids"]))
    print(batch["input_ids"].shape)
    print(type(batch["labels"]))
    print(batch["labels"].shape)
    print(type(batch["attention_mask"]))
    print(batch["attention_mask"].shape)

    tags, tag2id, id2tag = load_tag(label_type=DATASET_TYPE_NAME)
    args = ModelArguments()
    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_labels,
        finetuning_task=args.task_name,
        id2label=id2tag,
        label2id=tag2id,
    )
    model = RBERT.from_pretrained(args.model_name_or_path, config=config, args=args)
    output = model(**batch)
    print(type(output))
    print(output[0])
    print(output[1].shape)

    writer = SummaryWriter(log_dir='./log/' + time.strftime('%m-%d_%H_%M', time.localtime()))

    inputs = []
    inputs.append(batch["input_ids"])
    inputs.append(batch["attention_mask"])
    inputs.append(batch["token_type_ids"])
    inputs.append(batch["labels"])
    inputs.append(batch["e1_mask"])
    inputs.append(batch["e2_mask"])
    writer.add_graph(model, inputs)

    writer.close()
