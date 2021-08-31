#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : config.py
# @Author: sl
# @Date  : 2021/8/20 - 上午8:39


from dataclasses import dataclass, field

WORK_DIR = "/home/sl/workspace/python/a2020/ml-learn/nlp/re/rbert"

# albert: voidful/albert_chinese_base
# bert: bert-base-chinese
# roberta: hfl/chinese-bert-wwm-ext  hfl/chinese-roberta-wwm-ext
# DistilBert: adamlin/bert-distil-chinese
# xlnet: hfl/chinese-xlnet-base


BERT_PATH = "/home/sl/workspace/data/nlp/bert-base-chinese"
# BERT_PATH = "/home/sl/workspace/data/nlp/bert-ner"

DATA_DIR = "/home/sl/workspace/data/nlp/sts_1"
RELATION_DATA_DIR = DATA_DIR + "/rel.json"

# BERT_MODEL_NAME = "bert-base-chinese"
DATASET_TYPE_NAME = "semeval_2010_task8"
BERT_MODEL_NAME = BERT_PATH


@dataclass
class ModelArguments:
    dropout_rate: float = field(default=0.1, metadata={"help": "预训练模型输出向量表示的dropout"})
    num_labels: int = field(default=2, metadata={"help": "需要预测的标签数量"})
    model_name_or_path: str = field(default=BERT_PATH, metadata={"help": "BERT模型名称"})
    model_name: str = field(default="Bert", metadata={"help": "模型名称"})
    task_name: str = field(default="sts_1", metadata={"help": "任务名称"})
    pretrained_model_name: str = field(default="bert", metadata={"help": "预训练模型名称"})

    train_file: str = field(default=DATA_DIR + "/train.csv", metadata={"help": "训练数据的路径"})
    dev_file: str = field(default=DATA_DIR + "/dev.csv", metadata={"help": "测试数据的路径"})
    test_file: str = field(default=DATA_DIR + "/test.csv", metadata={"help": "测试数据的路径"})
    relation_file: str = field(default=DATA_DIR + "/rel.json", metadata={"help": "关系数据的路径"})

    checkpoint_dir: str = field(default=WORK_DIR + "/models/checkpoints", metadata={"help": "训练过程中的checkpoints的保存路径"})
    best_dir: str = field(default=WORK_DIR + "/models/best", metadata={"help": "最优模型的保存路径"})
    do_train: bool = field(default=True, metadata={"help": "是否进行训练"})
    do_eval: bool = field(default=True, metadata={"help": "是否在训练时进行评估"})
    do_predict: bool = field(default=True, metadata={"help": "是否在训练时进行预测"})
    no_cuda: bool = field(default=True, metadata={"help": "是否不用CUDA"})
    output: bool = field(default=True, metadata={"help": "是否输出"})
    overwrite_cache: bool = field(default=False, metadata={"help": "是否覆盖缓存"})
    debug: bool = field(default=True, metadata={"help": "是否正在调试"})

    epoch: int = field(default=5, metadata={"help": "训练的epoch"})
    train_batch_size: int = field(default=8, metadata={"help": "训练时的batch size"})
    eval_batch_size: int = field(default=8, metadata={"help": "评估时的batch size"})
    train_max_seq_length: int = field(default=64, metadata={"help": "训练时的最大长度"})
    eval_max_seq_length: int = field(default=64, metadata={"help": "评估时的最大长度"})

    bert_model_name: str = field(default=BERT_PATH, metadata={"help": "BERT模型名称"})
    output_dir: str = field(default=WORK_DIR + "/output/", metadata={"help": "输出路径"})
    eval_dir: str = field(default=WORK_DIR + "/eval/", metadata={"help": "输出路径"})
    result_save_name: str = field(default="result.json", metadata={"help": "输出文件"})

    eval_steps: int = field(default=4, metadata={"help": "每多少step进行验证一次"})
    save_steps: int = field(default=4, metadata={"help": "每多少step进行保存一次"})
    learning_rate: float = field(default=2e-5, metadata={"help": "The initial learning rate for Adam."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    warmup: float = field(default=0.05, metadata={"help": "Linear warmup over warmup_steps."})
    weight_decay: float = field(default=0.0, metadata={"help": "weight_decay."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Linear warmup over warmup_steps."})
    seed: int = field(default=42, metadata={"help": "seed"})
    num_relations: int = field(default=18, metadata={"help": "seed"})
