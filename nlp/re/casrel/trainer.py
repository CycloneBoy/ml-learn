#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : trainer.py
# @Author: sl
# @Date  : 2021/8/26 - 下午2:45
import json
import os
import time

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from nlp.re.casrel.data_loader import collate_fn
from nlp.re.casrel.model import CasRel
from nlp.re.casrel.utils import load_tag, load_tokenizer, logger, get_time_dif, get_checkpoint_dir


# logger = logging.getLogger(__name__)from nlp.re.rbert.model import RBERT


class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.tags, self.tag2id, self.id2tag = load_tag()
        self.num_labels = len(self.tags)

        self.config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=args.task_name,
            id2label=self.id2tag,
            label2id=self.tag2id,
        )
        self.model = CasRel.from_pretrained(args.model_name_or_path, config=self.config, args=args)
        self.tokenizer = load_tokenizer(args.model_name_or_path)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler,
                                      batch_size=self.args.train_batch_size,
                                      collate_fn=collate_fn)

        t_total = len(train_dataloader) * self.args.epoch
        num_warmup_steps = t_total * self.args.warmup

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total,
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Device = %s", self.device)
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.epoch)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Eval steps = %d", self.args.eval_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        start_time = time.time()

        train_result = []
        writer = SummaryWriter(log_dir='./log/' + time.strftime('%m-%d_%H_%M', time.localtime()))
        for epoch in range(self.args.epoch):
            epoch_start_time = time.time()
            logger.info('Epoch [{}/{}]'.format(epoch + 1, self.args.epoch))

            # pbar = ProgressBar(n_total=total_train_len, desc='Training')
            pbar = tqdm(total=len(train_dataloader), desc='Training-Batch[{}/{}]'.format(epoch + 1, self.args.epoch))
            for step, batch in enumerate(train_dataloader):
                self.model.train()

                for k, v in batch.items():
                    if k not in ["triples"]:
                        batch[k] = v.to(self.device)

                inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                          "sub_head": batch["sub_head"], "sub_tail": batch["sub_tail"],
                          "sub_heads": batch["sub_heads"], "sub_tails": batch["sub_tails"],
                          "obj_heads": batch["obj_heads"], "obj_tails": batch["obj_tails"]}

                optimizer.zero_grad()
                outputs = self.model(**inputs)
                loss = outputs[0]

                loss.backward()

                time_dif = get_time_dif(start_time)
                pbar.set_postfix({'loss': loss.item(), 'time': time_dif})
                pbar.update()

                tr_loss += loss.item()

                # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                global_step += 1

                # 每多少轮输出在训练集和验证集上的效果
                if global_step % self.args.eval_steps == 0:
                    # if global_step % len(train_dataloader) == 0:
                    metrics_result = self.evaluate("test")  # There is no dev set for semeval task
                    time_dif = get_time_dif(start_time)
                    msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, ' \
                          ' Val Loss: {3:>5.2},  Val F1: {4:>6.2%},  Val Acc: {5:>6.2%}, ' \
                          ' Val recall: {6:>6.2%},  Time: {7} '
                    logger.info(
                        msg.format(global_step, tr_loss / global_step, 0.0, metrics_result["loss"],
                                   metrics_result["f1"],
                                   metrics_result["precision"], metrics_result["recall"], time_dif))

                    metrics_result["total_batch"] = global_step
                    metrics_result["train_loss"] = tr_loss / global_step
                    metrics_result["train_time"] = time_dif
                    train_result.append(metrics_result)

                    writer.add_scalar("loss/train", tr_loss, global_step)
                    writer.add_scalar("loss/dev", metrics_result["loss"], global_step)
                    # writer.add_scalar("acc/train", train_acc, global_step)
                    writer.add_scalar("acc/dev", metrics_result["precision"], global_step)
                    writer.add_scalar("f1/dev", metrics_result["f1"], global_step)
                    writer.add_scalar("recall/dev", metrics_result["recall"], global_step)

                # 每多少轮保存一次
                if global_step % self.args.save_steps == 0:
                    self.save_model(global_step)
            logger.info('Epoch [{}/{}] finished, use time :{}'.format(epoch + 1, self.args.epoch,
                                                                      get_time_dif(epoch_start_time)))
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()

        logger.info("训练完毕")
        logger.info("Epoch	Training Loss	Validation Loss	F1	Precision	Recall  Use Time ")
        logger.info("-" * 100)
        for index, var in enumerate(train_result):
            msg = 'Iter: {0:>6},  Train Loss: {1:>5.2}, ' \
                  ' Val Loss: {2:>5.2},  Val F1: {3:>6.2%},  Val Acc: {4:>6.2%}, ' \
                  ' Val recall: {5:>6.2%},  Time: {6} '
            logger.info(msg.format(index + 1, var["train_loss"], var["loss"], var["f1"], var["precision"],
                                   var["recall"], var["train_time"]))

        writer.close()
        return global_step, tr_loss / global_step

    def evaluate(self, mode="test", h_bar=0.5, t_bar=0.5):
        # We use test dataset because semeval doesn't have dev dataset
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_dataloader = DataLoader(dataset, shuffle=False, batch_size=self.args.eval_batch_size,
                                     collate_fn=collate_fn)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        orders = ['subject', 'relation', 'object']
        correct_num, predict_num, gold_num = 0, 0, 0
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            for k, v in batch.items():
                if k not in ["triples"]:
                    batch[k] = v.to(self.device)

            inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"],
                      "sub_head": batch["sub_head"], "sub_tail": batch["sub_tail"],
                      "sub_heads": batch["sub_heads"], "sub_tails": batch["sub_tails"],
                      "obj_heads": batch["obj_heads"], "obj_tails": batch["obj_tails"]}

            with torch.no_grad():
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                encoded_text, outputs = self.model.get_encoded_text(input_ids, attention_mask)
                pred_sub_heads, pred_sub_tails = self.model.get_subs(encoded_text)

                sub_heads = torch.where(pred_sub_heads[0] > h_bar)[0]
                sub_tails = torch.where(pred_sub_tails[0] > t_bar)[0]
                subjects = []
                for sub_head in sub_heads:
                    sub_tail = sub_tails[sub_tails >= sub_head]
                    if len(sub_tail) > 0:
                        sub_tail = sub_tail[0]
                        subject = ''.join(self.tokenizer.decode(input_ids[0][sub_head: sub_tail + 1]).split())
                        subjects.append((subject, sub_head, sub_tail))

                if subjects:
                    triple_list = []
                    # [subject_num, seq_len, bert_dim]
                    repeated_encoded_text = encoded_text.repeat(len(subjects), 1, 1)
                    # [subject_num, 1, seq_len]
                    sub_head_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                                   device=self.device)
                    sub_tail_mapping = torch.zeros((len(subjects), 1, encoded_text.size(1)), dtype=torch.float,
                                                   device=self.device)
                    for subject_idx, subject in enumerate(subjects):
                        sub_head_mapping[subject_idx][0][subject[1]] = 1
                        sub_tail_mapping[subject_idx][0][subject[2]] = 1

                    sub_tail_mapping1 = sub_tail_mapping.to(repeated_encoded_text)
                    sub_head_mapping1 = sub_head_mapping.to(repeated_encoded_text)

                    pred_obj_heads, pred_obj_tails = self.model.get_objs_for_specific_sub(sub_head_mapping1,
                                                                                          sub_tail_mapping1,
                                                                                          repeated_encoded_text)
                    for subject_idx, subject in enumerate(subjects):
                        sub = subject[0]
                        obj_heads = torch.where(pred_obj_heads[subject_idx] > h_bar)
                        obj_tails = torch.where(pred_obj_tails[subject_idx] > t_bar)
                        for obj_head, rel_head in zip(*obj_heads):
                            for obj_tail, rel_tail in zip(*obj_tails):
                                if obj_head <= obj_tail and rel_head == rel_tail:
                                    rel = self.id2tag[int(rel_head)]
                                    obj = ''.join(self.tokenizer.decode(input_ids[0][obj_head: obj_tail + 1]).split())
                                    triple_list.append((sub, rel, obj))
                                    break

                    triple_set = set()
                    for s, r, o in triple_list:
                        triple_set.add((s, r, o))
                    pred_list = list(triple_set)

                else:
                    pred_list = []

                pred_triples = set(pred_list)
                gold_triples = set(self.to_tuple(batch['triples'][0]))
                correct_num += len(pred_triples & gold_triples)
                predict_num += len(pred_triples)
                gold_num += len(gold_triples)

                if self.args.output:
                    if not os.path.exists(self.args.output_dir):
                        os.mkdir(self.args.output_dir)
                    path = os.path.join(self.args.output_dir, self.args.result_save_name)
                    fw = open(path, 'w')
                    result = json.dumps({
                        'triple_list_gold': [
                            dict(zip(orders, triple)) for triple in gold_triples
                        ],
                        'triple_list_pred': [
                            dict(zip(orders, triple)) for triple in pred_triples
                        ],
                        'new': [
                            dict(zip(orders, triple)) for triple in pred_triples - gold_triples
                        ],
                        'lack': [
                            dict(zip(orders, triple)) for triple in gold_triples - pred_triples
                        ]
                    }, ensure_ascii=False)
                    fw.write(result + '\n')

        logger.info(
            "correct_num: {:3d}, predict_num: {:3d}, gold_num: {:3d}".format(correct_num, predict_num, gold_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        logger.info('f1: {:4.2f}, precision: {:4.2f}, recall: {:4.2f}'.format(f1_score, precision, recall))

        results = {}
        results["precision"] = precision
        results["acc"] = precision
        results["recall"] = recall
        results["f1"] = f1_score
        results["loss"] = 0

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  {} = {:.4f}".format(key, results[key]))

        return results

    def save_model(self, global_step):
        # Save model checkpoint (Overwrite)
        output_dir = os.path.join(self.args.output_dir, "checkpoint-{}".format(global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(output_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.tokenizer.save_vocabulary(output_dir)
        logger.info("")
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.output_dir):
            raise Exception("Model doesn't exists! Train first!")

        checkpoint_dir, max_step = get_checkpoint_dir(self.args.output_dir)

        self.args = torch.load(os.path.join(self.args.output_dir, "training_args.bin"))

        self.model = CasRel.from_pretrained(os.path.join(self.args.output_dir, checkpoint_dir), args=self.args)
        self.model.to(self.device)
        logger.info("***** Model Loaded *****")

    def to_tuple(self, triple_list):
        ret = []
        for triple in triple_list:
            ret.append((triple['subject'], triple['predicate'], triple['object']))
        return ret
