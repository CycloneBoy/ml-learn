#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: sl
# @Date  : 2021/4/15 -  下午10:31

"""
训练并评估模型
"""
from nlp.ner.evaluating import Metrics
from nlp.ner.model.crf import CRFModel
from nlp.ner.model.hmm import HMM
from util.nlp_utils import save_model, load_model


import os
from util.logger_utils import get_log
from util.nlp_utils import sent2features

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def hmm_train_eval(train_data, test_data, word2id, tag2id, remove_O=False):
    """
    训练并评估hmm模型
    :param train_data:
    :param test_data:
    :param word2id:
    :param tag2id:
    :param remove_O:
    :return:
    """
    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    hmm_mode = HMM(len(tag2id), len(word2id))
    log.info("开始训练HMM模型 - HMM模型: 状态数 - {}  观测数 - {}".format(len(tag2id), len(word2id)))
    hmm_mode.train(train_word_lists, train_tag_lists, word2id, tag2id)
    log.info("结束训练HMM模型,训练耗时:{}".format())
    # hmm_mode = load_model("./ckpts/hmm.pkl")

    save_model(hmm_mode, "./ckpts/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_mode.test(test_word_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def crf_train_eval(train_data,test_data,remove_O=False):
    """
    训练并评估CRF模型
    :param train_data:
    :param test_data:
    :param remove_O:
    :return:
    """

    train_word_lists, train_tag_lists = train_data
    test_word_lists, test_tag_lists = test_data

    crf_mode = CRFModel()
    crf_mode.train(train_word_lists, train_tag_lists)

    # crf_mode = load_model("./ckpts/crf.pkl")

    save_model(crf_mode, "./ckpts/crf.pkl")

    # 评估CRF模型
    pred_tag_lists = crf_mode.test(test_word_lists)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists