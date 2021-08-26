#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: sl
# @Date  : 2021/8/20 - 上午8:39


from dataclasses import dataclass, field

WORK_DIR = "/home/sl/workspace/python/a2020/ml-learn/nlp/re/rbert"
# BERT_PATH = "/home/sl/workspace/data/nlp/bert-base-chinese"
# BERT_PATH = "/home/sl/workspace/data/nlp/bert-ner"
BERT_PATH = "/home/sl/workspace/data/nlp/bert-base-uncased"
CLUENER_DATASET_DIR = "/home/sl/workspace/python/a2020/ml-learn/nlp/bertner/datasets/cluener"
SEMEVAL_TEST8_DATA_DIR = "/home/sl/workspace/python/a2020/ml-learn/data/nlp/SemevalTask8"
DATA_DIR = SEMEVAL_TEST8_DATA_DIR
# BERT_MODEL_NAME = "bert-base-chinese"
DATASET_TYPE_NAME = "semeval_2010_task8"
BERT_MODEL_NAME = BERT_PATH


@dataclass
class ModelArguments:
    dropout_rate: float = field(default=0.1, metadata={"help": "预训练模型输出向量表示的dropout"})
    num_labels: int = field(default=19, metadata={"help": "需要预测的标签数量"})
    model_name_or_path: str = field(default=BERT_PATH, metadata={"help": "BERT模型名称"})
    task_name: str = field(default="R_BERT", metadata={"help": "任务名称"})

    train_file: str = field(default=DATA_DIR + "/train.tsv", metadata={"help": "训练数据的路径"})
    dev_file: str = field(default=DATA_DIR + "/dev.tsv", metadata={"help": "测试数据的路径"})
    test_file: str = field(default=DATA_DIR + "/test.tsv", metadata={"help": "测试数据的路径"})

    checkpoint_dir: str = field(default=WORK_DIR + "/models/checkpoints", metadata={"help": "训练过程中的checkpoints的保存路径"})
    best_dir: str = field(default=WORK_DIR + "/models/best", metadata={"help": "最优模型的保存路径"})
    do_train: bool = field(default=True, metadata={"help": "是否进行训练"})
    do_eval: bool = field(default=True, metadata={"help": "是否在训练时进行评估"})
    do_predict: bool = field(default=True, metadata={"help": "是否在训练时进行预测"})
    no_cuda: bool = field(default=True, metadata={"help": "是否不用CUDA"})
    epoch: int = field(default=5, metadata={"help": "训练的epoch"})
    train_batch_size: int = field(default=8, metadata={"help": "训练时的batch size"})
    eval_batch_size: int = field(default=8, metadata={"help": "评估时的batch size"})
    bert_model_name: str = field(default=BERT_PATH, metadata={"help": "BERT模型名称"})
    output_dir: str = field(default=WORK_DIR + "/output/", metadata={"help": "输出路径"})
    eval_dir: str = field(default=WORK_DIR + "/eval/", metadata={"help": "输出路径"})

    eval_steps: int = field(default=4, metadata={"help": "每多少step进行验证一次"})
    save_steps: int = field(default=4, metadata={"help": "每多少step进行保存一次"})
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for Adam."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    warmup: float = field(default=0.05, metadata={"help": "Linear warmup over warmup_steps."})
    weight_decay: float = field(default=0.0, metadata={"help": "weight_decay."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Linear warmup over warmup_steps."})
    seed: int = field(default=42, metadata={"help": "seed"})


@dataclass
class OurTrainingArguments:
    checkpoint_dir: str = field(default=WORK_DIR + "/models/checkpoints", metadata={"help": "训练过程中的checkpoints的保存路径"})
    best_dir: str = field(default=WORK_DIR + "/models/best", metadata={"help": "最优模型的保存路径"})
    do_train: bool = field(default=True, metadata={"help": "是否进行训练"})
    do_eval: bool = field(default=True, metadata={"help": "是否在训练时进行评估"})
    do_predict: bool = field(default=True, metadata={"help": "是否在训练时进行预测"})
    no_cuda: bool = field(default=True, metadata={"help": "是否不用CUDA"})
    epoch: int = field(default=5, metadata={"help": "训练的epoch"})
    train_batch_size: int = field(default=8, metadata={"help": "训练时的batch size"})
    eval_batch_size: int = field(default=8, metadata={"help": "评估时的batch size"})
    bert_model_name: str = field(default=BERT_PATH, metadata={"help": "BERT模型名称"})
    output_dir: str = field(default=WORK_DIR + "/output/", metadata={"help": "输出路径"})
    eval_steps: int = field(default=4, metadata={"help": "每多少step进行验证一次"})
    save_steps: int = field(default=100, metadata={"help": "每多少step进行保存一次"})


@dataclass
class DataArguments:
    train_file: str = field(default=DATA_DIR + "/train.tsv", metadata={"help": "训练数据的路径"})
    dev_file: str = field(default=DATA_DIR + "/dev.tsv", metadata={"help": "测试数据的路径"})
    test_file: str = field(default=DATA_DIR + "/test.tsv", metadata={"help": "测试数据的路径"})
