#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/8/26 - 下午3:35
from nlp.re.rbert.config import ModelArguments
from nlp.re.rbert.data_loader import SemevalTask8Dataset, read_data
from nlp.re.rbert.trainer import Trainer
from nlp.re.rbert.utils import load_tokenizer, set_seed, check_file_exists


def main(args):
    set_seed(args)
    tokenizer = load_tokenizer(args.model_name_or_path)

    train_dataset = SemevalTask8Dataset(read_data(args.train_file), tokenizer=tokenizer)
    test_dataset = SemevalTask8Dataset(read_data(args.test_file), tokenizer=tokenizer)

    check_file_exists(args.output_dir + "/test.txt")
    check_file_exists(args.eval_dir + "/test.txt")
    trainer = Trainer(args, train_dataset=train_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")


if __name__ == '__main__':
    model_args = ModelArguments(save_steps=100)
    main(model_args)
