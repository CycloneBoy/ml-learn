#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_ner.py
# @Author: sl
# @Date  : 2021/8/18 - 下午9:05

import time

import torch
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


def train(config, train_dataset, model, tokenizer):
    """ Train the model """

    start_time = time.time()
    model.train()

    max_grad_norm = 1.0
    warmup = 0.05
    num_training_steps = len(train_dataset) * config.num_epochs
    num_warmup_steps = num_training_steps * warmup

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", len(train_dataset))
    print("  Num Epochs = %d", config.num_epochs)
    print("  Instantaneous batch size per GPU = %d", config.batch_size)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))

        for i, (trains, labels) in enumerate(train_dataset):
            model.train()
            outputs = model(trains)
            model.zero_grad()
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()  # 学习率衰减

            total_batch += 1

            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
            

def main():
    pass


if __name__ == '__main__':
    main()
