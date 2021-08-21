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

from data_utils import Example, read_data, load_tag
from data_utils import tokenizer, tag2id
from nlp.bert4ner.config import WORK_DIR, ModelArguments, BERT_PATH, BERT_MODEL_NAME, CLUENER_DATASET_DIR
from nlp.bert4ner.model import BertForNER, BertSpanForNER


class NERDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=128,
                 tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)):
        self.max_length = 512 if max_length > 512 else max_length
        self.texts = [torch.LongTensor(tokenizer.encode(example.text[: self.max_length - 2])) for example in examples]
        self.tag2id, self.id2tag = load_tag(label_type='bios')
        self.labels = []
        for example in examples:
            label = example.label
            label = [self.tag2id["<start>"]] + [self.tag2id[l] for l in label][: self.max_length - 2] + [
                self.tag2id["<eos>"]]
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


class NERSpanDataset(Dataset):
    def __init__(self, examples: List[Example], max_length=128,
                 tokenizer=BertTokenizer.from_pretrained(BERT_MODEL_NAME)):
        self.max_length = 512 if max_length > 512 else max_length
        self.texts = []
        self.tag2id, self.id2tag = load_tag(label_type='span')
        self.labels = []
        self.start_ids = []
        self.end_ids = []
        self.subjects_id = []
        self.input_len = []
        self.segment_ids = []

        for example in examples:
            textlist = example.text
            tokens = tokenizer.encode(textlist[: self.max_length - 2])

            subjects = example.subject
            start_ids = [0] * (len(tokens) - 2)
            end_ids = [0] * (len(tokens) - 2)
            segment_ids = [0] * len(tokens)
            input_len = [0] * len(tokens)
            input_len[0] = len(textlist)
            subjects_id = []

            for subject in subjects:
                label = subject[0]
                start = subject[1]
                end = subject[2]
                label_id = tag2id[label]
                start_ids[start] = label_id
                end_ids[end] = label_id
                subjects_id.append([label_id, start, end])

            # add sep token
            start_ids += [0]
            end_ids += [0]

            # add cls token
            start_ids = [0] + start_ids
            end_ids = [0] + end_ids

            self.texts.append(torch.LongTensor(tokens))
            self.start_ids.append(torch.LongTensor(start_ids))
            self.end_ids.append(torch.LongTensor(end_ids))
            self.subjects_id.append(torch.LongTensor(subjects_id))
            self.input_len.append(torch.LongTensor(input_len))
            self.segment_ids.append(torch.LongTensor(segment_ids))

        assert len(self.texts) == len(self.start_ids)
        for text, label in zip(self.texts, self.start_ids):
            assert len(text) == len(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        return {
            "input_ids": self.texts[item],
            "start_positions": self.start_ids[item],
            "end_positions": self.end_ids[item],
            "subjects_id": self.subjects_id[item],
            "input_len": self.input_len[item],
            "segment_ids": self.segment_ids[item]
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


def collate_fn_span(features) -> Dict[str, Tensor]:
    batch_input_ids = [feature["input_ids"] for feature in features]
    batch_start_positions = [feature["start_positions"] for feature in features]
    batch_end_positions = [feature["end_positions"] for feature in features]
    batch_subjects_id = [feature["subjects_id"] for feature in features]
    batch_input_len = [feature["input_len"] for feature in features]
    batch_segment_ids = [feature["segment_ids"] for feature in features]
    batch_attention_mask = [torch.ones_like(feature["input_ids"]) for feature in features]

    # padding
    batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    batch_start_positions = pad_sequence(batch_start_positions, batch_first=True, padding_value=0)
    batch_end_positions = pad_sequence(batch_end_positions, batch_first=True, padding_value=0)

    batch_attention_mask = pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
    assert batch_input_ids.shape == batch_start_positions.shape
    assert batch_input_ids.shape == batch_end_positions.shape
    assert batch_input_ids.shape == batch_attention_mask.shape

    return {"input_ids": batch_input_ids,
            "start_positions": batch_start_positions,
            "end_positions": batch_end_positions,
            "input_len": batch_input_len,
            "segment_ids": batch_segment_ids,
            "subjects_id": batch_subjects_id,
            "attention_mask": batch_attention_mask}


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


def ner_dev_metrics(preds, labels) -> Dict[str, float]:
    """
    该函数是回调函数，Trainer会在进行评估时调用该函数
    """
    # preds = eval_output.predictions
    # preds = np.argmax(preds, axis=-1).flatten()
    # labels = eval_output.label_ids.flatten()
    # labels为0表示为<pad>，因此计算时需要去掉该部分
    mask = labels != 0
    preds = preds[mask]
    labels = labels[mask]
    metrics = dict()
    metrics["f1"] = f1_score(labels, preds, average="macro")
    metrics["precision"] = precision_score(labels, preds, average="macro")
    metrics["recall"] = recall_score(labels, preds, average="macro")
    return metrics


def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S


def ner_span_metrics(eval_output: EvalPrediction) -> Dict[str, float]:
    """
    该函数是回调函数，Trainer会在进行评估时调用该函数
    """
    preds = eval_output.predictions
    start_logits, end_logits = preds
    preds = bert_extract_item(start_logits, end_logits)

    # preds = np.argmax(preds, axis=-1).flatten()
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


def ner_demo():
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


def span_demo():
    train_data = read_data(CLUENER_DATASET_DIR + "/train.json", "json")

    dataset = NERSpanDataset(train_data)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn_span)
    batch = next(iter(dataloader))
    print(batch)
    print(type(batch["input_ids"]))
    print(batch["input_ids"].shape)
    print(type(batch["start_positions"]))
    print(batch["start_positions"].shape)

    print(type(batch["end_positions"]))
    print(batch["end_positions"].shape)

    print(type(batch["attention_mask"]))
    print(batch["attention_mask"].shape)

    print("-" * 20)
    model_args = ModelArguments(use_lstm=False)
    model = BertSpanForNER.from_pretrained(BERT_PATH, model_args=model_args, output_hidden_states=False)
    output = model(**batch)
    print(output)

    print(type(output))
    print(output.loss)
    print(output.start_logits.shape)
    print(output.end_logits.shape)


if __name__ == "__main__":
    # span_demo()
    a1 = torch.rand((4, 30, 34))
    print(a1.shape)
    print(a1.reshape(-1, a1.shape[-1]).shape)
    print(a1.shape)
    a2 = torch.rand((4, 40, 34))

    a3 = a1.reshape(-1, a1.shape[-1])
    a4 = a2.reshape(-1, a2.shape[-1])

    print(a3.shape)
    print(a4.shape)

    b = torch.cat((a3, a4))
    print(b.shape)

    pass

    # writer = SummaryWriter(log_dir="runs/bert4ner_1")
    #
    # input = torch.rand(4, 128)
    # writer.add_graph(model, (input,))
    #
    # writer.close()
