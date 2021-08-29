#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/8/29 - 下午12:51
from nlp.re.casrel.config import ModelArguments
from nlp.re.casrel.data_loader import load_dataset
from nlp.re.casrel.trainer import Trainer
from nlp.re.casrel.utils import set_seed, load_tokenizer, check_file_exists, load_tag


def main(args):
    set_seed(args)
    tokenizer = load_tokenizer()

    train_dataset = load_dataset(args, tokenizer=tokenizer, data_type="train")
    test_dataset = load_dataset(args, tokenizer=tokenizer, data_type="dev")

    check_file_exists(args.output_dir + "/test.txt")
    check_file_exists(args.eval_dir + "/test.txt")
    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    tags, tag2id, id2tag = load_tag()
    model_args = ModelArguments(save_steps=100, num_relations=len(tags),
                                train_batch_size=4, eval_batch_size=1)
    main(model_args)
