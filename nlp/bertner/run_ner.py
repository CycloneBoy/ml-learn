#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_ner.py
# @Author: sl
# @Date  : 2021/8/18 - 下午9:05
import os
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertConfig

from nlp.bertner.models.bert_for_ner import BertSoftmaxForNer
from nlp.bertner.ner_metrics import SeqEntityScore
from nlp.bertner.processors.ner_seq import build_dataset, build_iterator
from nlp.bertner.progressbar import ProgressBar


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def check_file_exists(filename, delete=False):
    """检查文件是否存在"""
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print("文件夹不存在,创建目录:{}".format(dir_name))


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train_bert(config, model, train_iter, dev_iter, test_iter):
    """ Train the model """

    start_time = time.time()
    model.train()

    max_grad_norm = 1.0
    warmup = 0.05
    total_train_len = len(train_iter)
    num_training_steps = total_train_len * config.num_epochs
    num_warmup_steps = num_training_steps * warmup

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    # Train!
    print("***** Running training *****")
    print("  Num examples = %d", total_train_len)
    print("  Num Epochs = %d", config.num_epochs)
    print("  Instantaneous batch size per GPU = %d", config.batch_size)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升

    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        pbar = ProgressBar(n_total=total_train_len, desc='Training')
        for step, batch in enumerate(train_iter):
            model.train()
            inputs = batch_to_input(batch, config)

            outputs = model(**inputs)
            model.zero_grad()
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            loss.backward()
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()

            # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()  # 学习率衰减

            total_batch += 1

            # 每多少轮输出在训练集和验证集上的效果
            if total_batch % config.logging_steps == 0:
                # Log metrics
                print(" ")
                evaluate(config, model, dev_iter)

            # Save model checkpoint
            if total_batch % config.save_steps == 0:
                check_file_exists(config.save_path)
                torch.save(model.state_dict(), config.save_path)

                # Save model checkpoint
                output_dir = os.path.join(config.output_dir, "checkpoint-{}".format(total_batch))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Take care of distributed/parallel training
                model_to_save = (model.module if hasattr(model, "module") else model)
                model_to_save.save_pretrained(output_dir)
                torch.save(config, os.path.join(output_dir, "training_args.bin"))
                config.tokenizer.save_vocabulary(output_dir)
                print("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                print("Saving optimizer and scheduler states to %s", output_dir)

                model.train()

            time_dif = get_time_dif(start_time)
            print(f"进行第 {step} 次迭代，总共： {total_batch} ,  Time: {time_dif}")

    return total_batch, tr_loss / total_batch


def batch_to_input(batch, config):
    """小批量 转model input"""
    batch = tuple(t.to(config.device) for t in batch)
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
    if config.model_type != "distilbert":
        # XLM and RoBERTa don"t use segment_ids
        inputs["token_type_ids"] = (batch[2] if config.model_type in ["bert", "xlnet"] else None)
    return inputs


def evaluate(config, model, test_iter, prefix=""):
    metric = SeqEntityScore(config.id2label, markup=config.markup)
    eval_output_dir = config.output_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Eval!
    print("***** Running evaluation %s *****", prefix)
    print("  Num examples = %d", len(test_iter))
    print("  Batch size = %d", config.batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(test_iter), desc="Evaluating")
    for step, batch in enumerate(test_iter):
        model.eval()
        with torch.no_grad():
            inputs = batch_to_input(batch, config)
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
        out_label_ids = inputs['labels'].cpu().numpy().tolist()
        for i, label in enumerate(out_label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif out_label_ids[i][j] == config.label2id['[SEP]']:
                    metric.update(pred_paths=[temp_2], label_paths=[temp_1])
                    break
                else:
                    temp_1.append(config.id2label[out_label_ids[i][j]])
                    temp_2.append(preds[i][j])
        pbar(step)

    print(' ')
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    print("***** Eval results %s *****", prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
    print("***** Entity results %s *****", prefix)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)

    return results


class UserBertConfig(object):
    """配置参数"""

    def __init__(self, dataset="", name='bert-base-chinese'):
        self.model_name = 'bert_v1_' + name
        self.data_dir = "/home/sl/workspace/python/a2020/ml-learn/nlp/bertner/datasets/cluener"
        self.train_path = self.data_dir + '/data/train.txt'  # 训练集
        self.dev_path = self.data_dir + '/data/dev.txt'  # 验证集
        self.test_path = self.data_dir + '/data/test.txt'  # 测试集
        self.class_list = ["X", "B-address", "B-book", "B-company", 'B-game', 'B-government', 'B-movie', 'B-name',
                           'B-organization', 'B-position', 'B-scene', "I-address",
                           "I-book", "I-company", 'I-game', 'I-government', 'I-movie', 'I-name',
                           'I-organization', 'I-position', 'I-scene',
                           "S-address", "S-book", "S-company", 'S-game', 'S-government', 'S-movie',
                           'S-name', 'S-organization', 'S-position',
                           'S-scene', 'O', "[CLS]", "[SEP]"]
        self.output_dir = "."
        self.log_path = './log/' + self.model_name
        self.save_path = dataset + './model/' + self.model_name + '.ckpt'  # 模型训练结果
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.device = 'cpu'

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 4  # mini-batch大小
        self.pad_size = 128  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.hidden_dropout_prob = 0.5
        self.loss_type = "ce"
        self.id2label = {i: label for i, label in enumerate(self.class_list)}
        self.label2id = {label: i for i, label in enumerate(self.class_list)}

        self.hidden_size = 768
        self.pretrain_name = name
        # self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_name)
        # self.model_config = BertConfig.from_pretrained(self.pretrain_name, num_labels=self.num_classes,
        #                                                loss_type=self.loss_type,
        #                                                output_hidden_states=True)

        self.bert_path = "/home/sl/workspace/data/nlp/bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.model_config = BertConfig.from_pretrained(self.bert_path, num_labels=self.num_classes,
                                                       loss_type=self.loss_type,
                                                       output_hidden_states=True)

        self.train_max_seq_length = self.pad_size
        self.eval_max_seq_length = self.pad_size
        self.model_type = "bert"
        self.logging_steps = 2
        self.save_steps = 2
        # choices=['bios','bio']
        self.markup = 'bios'


def run_bert_ner():
    config = UserBertConfig()

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    print(f"device : {config.device}")
    print(f"model : {config.pretrain_name}")
    print(f"config : {config}")

    start_time = time.time()
    print("Loading data...")

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = BertSoftmaxForNer(config).to(config.device)
    init_network(model)
    print(model.parameters)
    print("Training/evaluation parameters %s", config)
    # train_bert(config, model, train_iter, dev_iter, test_iter)

    print(f"device : {config.device}")
    print(f"model : {config.pretrain_name}")
    print(f"config : {config}")

    start_time = time.time()
    print("Loading data...")
    train_data, dev_data, test_data = build_dataset(config)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    train_bert(config, model, train_iter, dev_iter, test_iter)


def main():
    run_bert_ner()


def test_eval():
    """验证数据准确性"""
    config = UserBertConfig()
    train_data, dev_data, test_data = build_dataset(config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = dev_iter

    metric = SeqEntityScore(config.id2label, markup=config.markup)

    # Eval!
    print("  Num examples = %d" % len(test_iter))
    print("  Batch size = %d" % config.batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(test_iter), desc="Evaluating")

    # for i, label in enumerate(out_label_ids):
    #     temp_1 = []
    #     temp_2 = []
    #     for j, m in enumerate(label):
    #         if j == 0:
    #             continue
    #         elif out_label_ids[i][j] == config.label2id['[SEP]']:
    #             metric.update(pred_paths=[temp_2], label_paths=[temp_1])
    #             break
    #         else:
    #             temp_1.append(config.id2label[out_label_ids[i][j]])
    #             temp_2.append(preds[i][j])

    print(' ')
    eval_loss = eval_loss / nb_eval_steps
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    results['loss'] = eval_loss
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)

    return results


if __name__ == '__main__':
    main()
