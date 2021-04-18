#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/4/15 -  下午10:37

"""
main
"""
from nlp.ner.data import build_corpus
from nlp.ner.evaluate import hmm_train_eval, crf_train_eval

import os
from util.logger_utils import get_log
from util.nlp_utils import sent2features

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def main():
    """
    训练模型，评估结果
    :return:
    """
    # 读取数据
    log.info("开始读取数据")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)
    log.info("结束读取数据")

    # # 训练评估hmm模型
    # print("正在训练评估HMM模型")
    # hmm_pred= hmm_train_eval((train_word_lists, train_tag_lists),
    #                (test_word_lists, test_tag_lists),
    #                word2id,tag2id)

    # 训练评估CRF模型
    log.info("开始训练评估CRF模型")
    crf_pred= crf_train_eval((train_word_lists, train_tag_lists),
                   (test_word_lists, test_tag_lists))
    pass


if __name__ == '__main__':
    main()
