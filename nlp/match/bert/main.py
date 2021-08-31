#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/8/30 - 下午10:36
from nlp.match.bert.config import ModelArguments
from nlp.match.bert.data_loader import load_dataset
from nlp.match.bert.trainer import Trainer
from nlp.match.bert.utils import load_tokenizer, check_file_exists, load_tag


def main(args):
    tokenizer = load_tokenizer(args=args)

    test_dataset = load_dataset(args, tokenizer=tokenizer, data_type="test")
    train_dataset = load_dataset(args, tokenizer=tokenizer, data_type="train")
    dev_dataset = load_dataset(args, tokenizer=tokenizer, data_type="dev")
    # test_dataset = load_dataset(args, tokenizer=tokenizer, data_type="test")

    check_file_exists(args.output_dir + "/test.txt")
    check_file_exists(args.eval_dir + "/test.txt")
    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    tags, tag2id, id2tag = load_tag()
    model_args = ModelArguments(save_steps=100, num_relations=len(tags),
                                train_batch_size=64, eval_batch_size=1, debug=True,
                                model_name='esim')
    main(model_args)
