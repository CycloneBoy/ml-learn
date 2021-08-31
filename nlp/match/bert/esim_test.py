#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : esim_test.py
# @Author: sl
# @Date  : 2021/8/31 - 下午5:20
from torch.utils.data.dataloader import DataLoader

from nlp.match.bert.config import ModelArguments
from nlp.match.bert.data_loader import load_dataset, collate_fn_rnn
from nlp.match.bert.model import ESIM
from nlp.match.bert.utils import load_vocab

if __name__ == '__main__':
    args = ModelArguments(save_steps=100, model_name="rnn")

    word2idx, idx2word, vocab = load_vocab(args.vocab_file)

    train_dataset = load_dataset(args, data_type="test")
    print(train_dataset[0])

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=64, collate_fn=collate_fn_rnn)

    batch = next(iter(train_dataloader))
    print(batch.keys())
    print(type(batch["text_a_token"]))
    print(batch["text_a_token"].shape)

    print(type(batch["text_a_length"].shape))
    print(batch["text_a_length"])

    print(type(batch["text_b_token"]))
    print(batch["text_b_token"].shape)

    print(type(batch["text_b_length"].shape))
    print(batch["text_b_length"])

    print(type(batch["labels"]))
    print(batch["labels"].shape)

    model = ESIM(hidden_size=200, embeddings=None, dropout=0.2, num_labels=2, device='cpu')

    inputs = {"q1": batch["text_a_token"], "q1_lengths": batch["text_a_length"],
              "q2": batch["text_b_token"], "q2_lengths": batch["text_b_length"],
              "labels": batch["labels"]}

    output = model(**inputs)
    print(type(output))
    print(output[0])
    print(type(output[0]))
    print(output[1])
    print(output[1].shape)

    print(output[2])
    print(output[2].shape)

    probs = output[2]
    _, out_classes = probs.max(dim=1)
    correct = (out_classes == batch["labels"]).sum()
    print(correct.item())
