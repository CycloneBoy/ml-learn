#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_utils.py
# @Author: sl
# @Date  : 2021/8/20 - 上午8:47
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers.trainer_utils import EvalPrediction

from data_utils import Example, read_data
from data_utils import tokenizer, tag2id
from nlp.bert4ner.config import WORK_DIR, ModelArguments, BERT_PATH, BERT_MODEL_NAME
from nlp.bert4ner.model import BertForNER


class NERDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=128,
                 tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)):
        self.max_length = 512 if max_length > 512 else max_length
        self.texts = [torch.LongTensor(tokenizer.encode(example.text[: self.max_length - 2])) for example in examples]
        self.labels = []
        for example in examples:
            label = example.label
            label = [tag2id["<start>"]] + [tag2id[l] for l in label][: self.max_length - 2] + [tag2id["<eos>"]]
            self.labels.append(torch.LongTensor(label))
        assert len(self.texts) == len(self.labels)
        for text, label in zip(self.texts, self.labels):
            assert len(text) == len(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "input_ids": self.texts[item],
            "labels": self.labels[item]
        }


def collate_fn(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_labels = [feature["labels"] for feature in features]
    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]
    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_labels = pad_sequence(batch_labels, batch_first=True, padding_value=tag2id["<pad>"])
    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    assert batch_input_ids.shape == batch_labels.shape
    return {"input_ids": batch_input_ids, "labels": batch_labels, "attention_mask": batch_attention_mask}


def ner_metrics(eval_output: EvalPrediction) -> Dict[str, float]:
    """
    该函数是回调函数，Trainer会在进行评估时调用该函数
    """
    preds = eval_output.predictions
    preds = np.argmax(preds, axis=-1).flatten()
    labels = eval_output.label_ids.flatten()
    # labels为0表示为<pad>，因此计算时需要去掉该部分
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average="macro")
    metrics["precision"] = precision_score(labels, preds, average="macro")
    metrics["recall"] = recall_score(labels, preds, average="macro")
    return metrics


if __name__ == "__main__":
    pass

    train_data = read_data(WORK_DIR + "/data/train.txt")
    eval_data = read_data(WORK_DIR + "/data/dev.txt")

    dataset = NERDataset(train_data)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(batch)
    print(type(batch["input_ids"]))
    print(batch["input_ids"].shape)
    print(type(batch["labels"]))
    print(batch["labels"].shape)
    print(type(batch["attention_mask"]))
    print(batch["attention_mask"].shape)

    print("-" * 20)
    model_args = ModelArguments(use_lstm=False)
    model = BertForNER.from_pretrained(BERT_PATH, model_args=model_args, output_hidden_states=False)
    output = model(**batch)
    print(output)

    print(type(output))
    print(output.loss)
    print(output.logits.shape)

    # writer = SummaryWriter(log_dir="runs/bert4ner_1")
    #
    # input = torch.rand(4, 128)
    # writer.add_graph(model, (input,))
    #
    # writer.close()
