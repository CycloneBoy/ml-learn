#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py
# @Author: sl
# @Date  : 2021/4/15 -  下午10:31

"""
训练并评估模型
"""
import time
from collections import Counter

from nlp.ner.evaluating import Metrics
from nlp.ner.model.blistm_crf import BiLSTM_Model
from nlp.ner.model.crf import CRFModel
from nlp.ner.model.hmm import HMM
from util.nlp_utils import save_model, load_model, flatten_lists

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
    # log.info("结束训练HMM模型,训练耗时:{}".format())
    # hmm_mode = load_model("./ckpts/hmm.pkl")

    save_model(hmm_mode, "./ckpts/hmm.pkl")

    # 评估hmm模型
    pred_tag_lists = hmm_mode.test(test_word_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def crf_train_eval(train_data, test_data, remove_O=False):
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


def bilstm_train_and_eval(train_data, dev_data, test_data,
                          word2id, tag2id, crf=True, remove_O=False):
    """
    训练并评估BiLSTM模型
    :param train_data:
    :param dev_data:
    :param test_data:
    :param word2id:
    :param tag2id:
    :param crf:
    :param remove_O:
    :return:
    """

    train_word_lists, train_tag_lists = train_data
    dev_word_lists, dev_tag_lists = dev_data
    test_word_lists, test_tag_lists = test_data

    start = time.time()
    vocab_size = len(word2id)
    out_size = len(tag2id)

    bilstm_model = BiLSTM_Model(vocab_size, out_size, crf=crf)
    bilstm_model.train(train_word_lists, train_tag_lists,
                       dev_word_lists, dev_tag_lists, word2id, tag2id)

    # crf_mode = load_model("./ckpts/crf.pkl")

    model_name = 'bilstm_crf' if crf else 'bilstm'
    save_model(bilstm_model, "./ckpts/{}.pkl".format(model_name))

    print("训练完毕,共用时{}秒.".format(int(time.time() - start)))
    print("评估{}模型中...".format(model_name))

    # 评估CRF模型
    pred_tag_lists, test_tag_lists = bilstm_model.test \
        (test_word_lists, test_tag_lists, word2id, tag2id)

    metrics = Metrics(test_tag_lists, pred_tag_lists, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tag_lists


def ensemble_evaluate(results, targets, remove_O=False):
    """
    ensemble多个模型
    :param results:
    :param targets:
    :param remove_O:
    :return:
    """
    for i in range(len(results)):
        results[i] = flatten_lists(results[i])

    pred_tags = []
    for result in zip(*results):
        ensemble_tag = Counter(result).most_common(1)[0][0]
        pred_tags.append(ensemble_tag)

    targets = flatten_lists(targets)
    assert len(pred_tags) == len(targets)

    print("Ensemble 四个模型的结果如下：")
    metrics = Metrics(targets, pred_tags, remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()

    return pred_tags
