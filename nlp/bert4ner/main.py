#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/8/21 - 下午2:27
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, BertTokenizer, AdamW, get_linear_schedule_with_warmup, EvalPrediction
from transformers.file_utils import logger, logging

from config import ModelArguments, OurTrainingArguments, DataArguments, CLUENER_DATASET_DIR
from data_utils import read_data, load_tag
from model import AlbertSpanForNER
from nlp.bert4ner.common import json_to_text
from nlp.bertner.ner_metrics import SeqEntityScore, compute_metric, SpanEntityScore
from train_utils import ner_metrics, bert_extract_item, NERSpanDataset, collate_fn_span

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
    # train_dataset = NERDataset(read_data(data_args.train_file, data_type="json"), tokenizer=tokenizer)
    # eval_dataset = NERDataset(read_data(data_args.dev_file, data_type="json"), tokenizer=tokenizer)

    train_dataset = NERSpanDataset(read_data(data_args.train_file, data_type="json"), tokenizer=tokenizer)
    eval_dataset = NERSpanDataset(read_data(data_args.dev_file, data_type="json"), tokenizer=tokenizer)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=training_args.train_batch_size,
                                  collate_fn=collate_fn_span)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=training_args.eval_batch_size,
                                 collate_fn=collate_fn_span)

    # 加载预训练模型 "bert-base-chinese"
    # model = BertForNER.from_pretrained(args.bert_model_name, model_args=model_args)
    # model = BertCrfForNER.from_pretrained(args.bert_model_name, model_args=model_args)
    # model = BertSpanForNER2.from_pretrained(args.bert_model_name, model_args=model_args)
    model = AlbertSpanForNER.from_pretrained(args.bert_model_name, model_args=model_args)
    model.to(device)

    if args.do_train:
        print("进行训练")
        global_step, tr_loss = train(model, args, train_dataloader, eval_dataloader, tokenizer, prefix="train")
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        save_model(model, args, tokenizer)
        print("训练完毕")

    if args.do_eval:
        print("进行验证")
        evaluate(model, eval_dataloader, prefix='finish', ner_type='span')
        # torch.save(model.state_dict(), f"./bert_{total_batch}.ptl")
        print("验证完毕")

    if args.do_predict:
        print("进行预测")
        predict(model, args, eval_dataloader, 'finish', )
        # torch.save(model.state_dict(), f"./bert_{total_batch}.ptl")
        print("预测完毕")


def train(model, args, train_dataloader, eval_dataloader, tokenizer, prefix=""):
    start_time = time.time()
    max_grad_norm = 1.0
    warmup = 0.05
    learning_rate = 5e-5

    total_train_len = len(train_dataloader)
    num_training_steps = total_train_len * args.epoch
    num_warmup_steps = num_training_steps * warmup

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # Train!
    print("***** Running training *****")
    print("  Num examples = %d" % total_train_len)
    print("  Num Epochs = %d" % args.epoch)
    print("  Instantaneous batch size per GPU = %d" % args.train_batch_size)

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

        # pbar = ProgressBar(n_total=total_train_len, desc='Training')
        pbar = tqdm(total=total_train_len, desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            for k, v in batch.items():
                if k not in ["input_len", "subjects_id"]:
                    batch[k] = v.to(device)

            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                      "start_positions": batch["start_positions"], "end_positions": batch["end_positions"],
                      "token_type_ids": batch["segment_ids"]}

            optimizer.zero_grad()
            outputs = model(**inputs)

            loss = outputs[0]

            loss.backward()
            # pbar(step, {'loss': loss.item()})
            time_dif = get_time_dif(start_time)
            pbar.set_postfix({'loss': loss.item(), 'time': time_dif})
            pbar.update()
            tr_loss += loss.item()

            # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

            total_batch += 1

            # print(f"进行第 {step} 次迭代，总共： {total_batch} ,  Time: {time_dif}")

            # 每多少轮输出在训练集和验证集上的效果
            # if total_batch % args.eval_steps == 0:
            if total_batch % total_train_len == 0:
                # Log metrics
                print("进行第 %d 次评估" % total_batch)
                prefix = "{}".format(total_batch)
                metrics_result = evaluate(model, eval_dataloader, prefix, 'span')
                dev_loss = metrics_result["loss"]
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # torch.save(model.state_dict(), f"./bert_{total_batch}.ptl")
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

            # 每多少轮保存一次
            if total_batch % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(total_batch))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                tokenizer.save_vocabulary(output_dir)

                logger.info("Saving model checkpoint to %s", output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        print('Epoch [{}/{}] finished, use time :{}'.format(epoch + 1, args.epoch, get_time_dif(epoch_start_time)))
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    print("训练完毕")
    print("Epoch	Training Loss	Validation Loss	F1	Precision	Recall  Use Time ")
    print("-" * 100)
    for index, var in enumerate(train_result):
        msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}, ' \
              ' Val Loss: {2:>5.2},  Val F1: {3:>6.2%},  Val Acc: {4:>6.2%}, ' \
              ' Val recall: {5:>6.2%},  Time: {6} '
        print(msg.format(index + 1, var["train_loss"], var["loss"], var["f1"], var["precision"],
                         var["recall"], var["train_time"]))

    return total_batch, tr_loss / total_batch


def evaluate(model, eval_dataloader, prefix="", ner_type="crf"):
    if ner_type == 'crf':
        return evaluate_crf(model, eval_dataloader, prefix=prefix)
    else:
        return evaluate_span(model, eval_dataloader, prefix=prefix)


def evaluate_crf(model, eval_dataloader, prefix=""):
    """评估"""

    # Eval!
    print("***** Running evaluation %s *****" % prefix)
    print("  Num examples = %d" % len(eval_dataloader))

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    predict_t = []
    predict_result = []

    tag2id, id2tag = load_tag(label_type='bios')
    metric = SeqEntityScore(id2tag, markup='bios')

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

            # 保存
            result = {}
            result["id"] = step + 1
            result["input_ids"] = inputs['input_ids'].cpu().numpy()
            result["labels"] = inputs['labels'].cpu().numpy()
            result["attention_mask"] = inputs['attention_mask'].cpu().numpy()
            result["predict_labels"] = np.argmax(logits.cpu().numpy(), axis=2)
            predict_result.append(result)

            # 计算metric
            preds = predict.tolist()
            out_label_ids = labels.tolist()
            compute_metric(metric, preds, out_label_ids, tag2id, id2tag)

            nb_eval_steps += 1
            pbar(step)
            if step == 4:
                break

    eval_loss = eval_loss / nb_eval_steps

    # save_pickle(predict_result, "./predict_result_{}.pkl".format(prefix))

    # 第一种计算指标的方式
    predict_all_2 = predict_t[0]
    for i in range(1, len(predict_t)):
        predict_all_2 = torch.cat((predict_all_2, predict_t[i]))

    eval_data = EvalPrediction(predictions=predict_all_2.numpy(), label_ids=labels_all)
    results = ner_metrics(eval_data)
    results['loss'] = eval_loss

    print("第一种计算指标")
    print("***** Eval results  %s *****" % prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
    print(info)

    print("第二种计算指标")
    eval_info, entity_info = metric.result()
    results2 = {f'{key}': value for key, value in eval_info.items()}
    results2['loss'] = eval_loss
    results2["precision"] = results2["acc"]

    print("***** Eval results %s *****" % prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results2.items()])
    print(info)
    print("***** Entity results %s *****" % prefix)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)

    return results2


def evaluate_span(model, eval_dataloader, prefix=""):
    """评估"""

    # Eval!
    print("***** Running evaluation %s *****" % prefix)
    print("  Num examples = %d" % len(eval_dataloader))

    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_dataloader), desc="Evaluating")

    tag2id, id2tag = load_tag(label_type='span')
    metric = SpanEntityScore(id2tag)

    for step, batch in enumerate(eval_dataloader):
        for k, v in batch.items():
            if k not in ["input_len", "subjects_id"]:
                batch[k] = v.to(device)

        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                      "start_positions": batch["start_positions"], "end_positions": batch["end_positions"],
                      "token_type_ids": batch["segment_ids"]}

            outputs = model(**inputs)
        tmp_eval_loss, start_logits, end_logits = outputs[:3]
        eval_loss += tmp_eval_loss.item()

        R = bert_extract_item(start_logits, end_logits)
        result_t = []
        for sub in batch['subjects_id']:
            result_t.extend(sub)
        T = result_t
        metric.update(true_subject=T, pred_subject=R)

        nb_eval_steps += 1
        pbar(step)
        if step == 4:
            break

    eval_loss = eval_loss / nb_eval_steps

    # save_pickle(predict_result, "./predict_result_{}.pkl".format(prefix))

    print("第二种计算指标")
    eval_info, entity_info = metric.result()
    results2 = {f'{key}': value for key, value in eval_info.items()}
    results2['loss'] = eval_loss
    results2["precision"] = results2["acc"]

    print("***** Eval results %s *****" % prefix)
    info = "-".join([f' {key}: {value:.4f} ' for key, value in results2.items()])
    print(info)
    print("***** Entity results %s *****" % prefix)
    for key in sorted(entity_info.keys()):
        print("******* %s results ********" % key)
        info = "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
        print(info)

    return results2


def predict(model, args, test_dataloader, prefix=""):
    """预测"""

    pred_output_dir = args.output_dir
    if not os.path.exists(pred_output_dir):
        os.makedirs(pred_output_dir)

    # prediction!
    print("***** Running prediction %s *****" % prefix)
    print("  Num examples = %d" % len(test_dataloader))
    print("  Batch size = %d" % args.eval_batch_size)

    results = []
    pbar = ProgressBar(n_total=len(test_dataloader), desc="Predicting")
    tag2id, id2tag = load_tag(label_type='bios')

    for step, batch in enumerate(test_dataloader):
        for k, v in batch.items():
            if k not in ["input_len", "subjects_id"]:
                batch[k] = v.to(device)

        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                      "start_positions": batch["start_positions"], "end_positions": batch["end_positions"],
                      "token_type_ids": batch["segment_ids"]}

            outputs = model(**inputs)

        start_logits, end_logits = outputs[:2]
        R = bert_extract_item(start_logits, end_logits)

        if R:
            label_entities = [[id2tag[x[0]], x[1], x[2]] for x in R]
        else:
            label_entities = []

        json_d = {}
        json_d['id'] = step
        json_d['entities'] = label_entities
        results.append(json_d)
        pbar(step)

    print(" ")
    output_predic_file = os.path.join(pred_output_dir, prefix, "test_prediction.json")
    output_submit_file = os.path.join(pred_output_dir, prefix, "test_submit.json")
    with open(output_predic_file, "w") as writer:
        for record in results:
            writer.write(json.dumps(record) + '\n')
    test_text = []
    with open(os.path.join(CLUENER_DATASET_DIR, "test.json"), 'r') as fr:
        for line in fr:
            test_text.append(json.loads(line))
    test_submit = []
    for x, y in zip(test_text, results):
        json_d = {}
        json_d['id'] = x['id']
        json_d['label'] = {}
        entities = y['entities']
        words = list(x['text'])
        if len(entities) != 0:
            for subject in entities:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                word = "".join(words[start:end + 1])
                if tag in json_d['label']:
                    if word in json_d['label'][tag]:
                        json_d['label'][tag][word].append([start, end])
                    else:
                        json_d['label'][tag][word] = [[start, end]]
                else:
                    json_d['label'][tag] = {}
                    json_d['label'][tag][word] = [[start, end]]
        test_submit.append(json_d)
    json_to_text(output_submit_file, test_submit)

    return results


def save_model(model, args, tokenizer):
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_vocabulary(args.output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


if __name__ == '__main__':
    model_args = ModelArguments(use_lstm=False, ner_num_labels=11)
    data_args = DataArguments()
    training_args = OurTrainingArguments(train_batch_size=64, eval_batch_size=24)
    run(model_args, data_args, training_args)
