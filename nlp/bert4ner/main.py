#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/8/21 - 下午2:27
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments, BertTokenizer, AdamW, get_linear_schedule_with_warmup, EvalPrediction
from transformers.file_utils import logger, logging

from config import ModelArguments, OurTrainingArguments, DataArguments
from data_utils import read_data
from model import BertForNER
from train_utils import NERDataset, collate_fn, ner_metrics

logger.setLevel(logging.INFO)

device = "cpu"


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ProgressBar(object):
    '''
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='training')
        >>> step = 2
        >>> pbar(step=step)
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def batch_to_input(batch):
    """小批量 转model input"""
    batch = tuple(t.to(device) for t in batch)
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    return inputs


def run(model_args: ModelArguments, data_args: DataArguments, args: OurTrainingArguments):
    # 设定训练参数
    training_args = TrainingArguments(output_dir=args.checkpoint_dir,  # 训练中的checkpoint保存的位置
                                      num_train_epochs=args.epoch,
                                      do_eval=args.do_eval,  # 是否进行评估
                                      do_predict=args.do_predict,  # 是否进行预测
                                      evaluation_strategy="epoch",  # 每个epoch结束后进行评估
                                      per_device_train_batch_size=args.train_batch_size,
                                      per_device_eval_batch_size=args.eval_batch_size,
                                      load_best_model_at_end=True,  # 训练完成后加载最优模型
                                      no_cuda=args.no_cuda,
                                      metric_for_best_model="f1"  # 评估最优模型的指标，该指标是ner_metrics返回评估指标中的key
                                      )
    # 构建分词器
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)

    # 构建dataset
    train_dataset = NERDataset(read_data(data_args.train_file, data_type="json"), tokenizer=tokenizer)
    eval_dataset = NERDataset(read_data(data_args.dev_file, data_type="json"), tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=training_args.train_batch_size,
                                  collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=training_args.train_batch_size,
                                 collate_fn=collate_fn)

    # 加载预训练模型 "bert-base-chinese"
    model = BertForNER.from_pretrained(args.bert_model_name, model_args=model_args)
    model.to(device)

    start_time = time.time()
    model.train()

    max_grad_norm = 1.0
    warmup = 0.05
    total_train_len = len(train_dataloader)
    num_training_steps = total_train_len * args.epoch
    num_warmup_steps = num_training_steps * warmup

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    # Train!
    print("***** Running training *****")
    print("  Num examples = %d" % total_train_len)
    print("  Num Epochs = %d" % args.epoch)
    print("  Instantaneous batch size per GPU = %d", args.train_batch_size)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    train_result = []
    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        print('Epoch [{}/{}]'.format(epoch + 1, args.epoch))
        pbar = ProgressBar(n_total=total_train_len, desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            for k, v in batch.items():
                batch[k] = v.to(device)
            inputs = batch

            optimizer.zero_grad()
            outputs = model(**inputs)
            model.zero_grad()
            loss = outputs[0]

            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()

            # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            total_batch += 1
            time_dif = get_time_dif(start_time)
            print(f"进行第 {step} 次迭代，总共： {total_batch} ,  Time: {time_dif}")

            # 每多少轮输出在训练集和验证集上的效果
            if total_batch % 4 == 0:
                # Log metrics
                print("进行第 %d 次评估" % total_batch)
                metrics_result = evaluate(model, eval_dataloader)
                dev_loss = metrics_result["loss"]
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), f"./bert_{total_batch}.ptl")
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, ' \
                      ' Val Loss: {3:>5.2},  Val F1: {4:>6.2%},  Val Acc: {5:>6.2%}, ' \
                      ' Val recall: {6:>6.2%},  Time: {7} , improve：  {8}'
                print(msg.format(total_batch, tr_loss / total_batch, 0.0, dev_loss, metrics_result["f1"],
                                 metrics_result["precision"],
                                 metrics_result["recall"], time_dif, improve))

                metrics_result["total_batch"] = total_batch
                metrics_result["train_loss"] = tr_loss / total_batch
                metrics_result["train_time"] = time_dif
                train_result.append(metrics_result)

        print('Epoch [{}/{}] finished, use time :{}'.format(epoch + 1, args.epoch, get_time_dif(epoch_start_time)))

    print("训练完毕")
    evaluate(model, eval_dataloader)
    torch.save(model.state_dict(), f"./bert_{total_batch}.ptl")

    print("Epoch	Training Loss	Validation Loss	F1	Precision	Recall  Use Time ")
    print("-" * 100)
    for index, var in enumerate(train_result):
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}, ' \
              ' Val Loss: {2:>5.2},  Val F1: {3:>6.2%},  Val Acc: {4:>6.2%}, ' \
              ' Val recall: {5:>6.2%},  Time: {6} '
        print(msg.format(index + 1, var["train_loss"], var["loss"], var["f1"], var["precision"],
                         var["recall"], var["train_time"]))

    return total_batch, tr_loss / total_batch


def evaluate(model, eval_dataloader):
    """评估"""

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = %d" % len(eval_dataloader))

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    predict_t = []

    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            inputs = batch
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()

            labels = inputs['labels'].cpu().numpy()
            predict = np.argmax(logits.cpu().numpy(), axis=2)

            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

            predict_test = logits.cpu()
            predict_t.append(predict_test.reshape(-1, predict_test.shape[-1]))

            nb_eval_steps += 1
            pbar(step)
            # if step == 4:
            #     break

    eval_loss = eval_loss / nb_eval_steps

    predict_all_2 = predict_t[0]
    for i in range(1, len(predict_t)):
        predict_all_2 = torch.cat((predict_all_2, predict_t[i]))

    eval_data = EvalPrediction(predictions=predict_all_2.numpy(), label_ids=labels_all)
    results = ner_metrics(eval_data)
    results['loss'] = eval_loss

    print("")
    print("***** Eval results  *****")
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)

    return results


if __name__ == '__main__':
    model_args = ModelArguments(use_lstm=False)
    data_args = DataArguments()
    training_args = OurTrainingArguments(train_batch_size=4, eval_batch_size=4)
    run(model_args, data_args, training_args)
