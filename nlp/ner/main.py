#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : main.py
# @Author: sl
# @Date  : 2021/4/15 -  下午10:37

"""
main
"""
from nlp.ner.data import build_corpus
from nlp.ner.evaluate import hmm_train_eval, crf_train_eval, bilstm_train_and_eval, ensemble_evaluate

import os
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf

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
    # hmm_pred = hmm_train_eval((train_word_lists, train_tag_lists),
    #                           (test_word_lists, test_tag_lists),
    #                           word2id, tag2id)
    #
    # # 训练评估CRF模型
    # log.info("开始训练评估CRF模型")
    # crf_pred = crf_train_eval((train_word_lists, train_tag_lists),
    #                           (test_word_lists, test_tag_lists))

    # 训练评估BI-LSTM模型
    # for_crf = True
    # print("正在训练评估双向LSTM模型,是否使用crf: {} ...".format(for_crf))
    # bilstm_word2id, bilstm_tag2id = extend_maps(word2id, tag2id, for_crf=for_crf)
    #
    # lstm_pred = bilstm_train_and_eval(
    #     (train_word_lists, train_tag_lists),
    #     (dev_word_lists, dev_tag_lists),
    #     (test_word_lists, test_tag_lists),
    #     bilstm_word2id, bilstm_tag2id,
    #     crf=for_crf
    # )

    print("正在训练评估Bi-LSTM+CRF模型...")
    # 如果是加了CRF的lstm还要加入<start>和<end> (解码的时候需要用到)
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)

    # 还需要额外的一些数据处理
    train_word_lists, train_tag_lists = process_data_for_lstmcrf(train_word_lists, train_tag_lists)
    dev_word_lists, dev_tag_lists = process_data_for_lstmcrf(dev_word_lists, dev_tag_lists)
    test_word_lists, test_tag_lists = process_data_for_lstmcrf(test_word_lists, test_tag_lists, test=True)

    lstmcrf_pred = bilstm_train_and_eval(
        (train_word_lists, train_tag_lists),
        (dev_word_lists, dev_tag_lists),
        (test_word_lists, test_tag_lists),
        crf_word2id, crf_tag2id
    )

    # ensemble_evaluate([hmm_pred, crf_pred,  lstmcrf_pred],
    #                   test_tag_lists)


if __name__ == '__main__':
    main()
