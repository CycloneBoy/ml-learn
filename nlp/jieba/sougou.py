#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : sougou.py
# @Author: sl
# @Date  : 2021/4/7 -  下午10:06
import glob
import os
import random
import re
from datetime import datetime, timedelta

import jieba
from dateutil.parser import parse

from util.common_utils import get_TF
from util.file_utils import get_news_path, get_content
from util.logger_utils import get_log
import os

from util.nlp_utils import stop_words

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py",'')))


def test_jieba():
    sent = "中文分词是文本处理不可或缺的一部分"
    seg_list = jieba.cut(sent, cut_all=True)
    log.info("全模式:{}".format('/'.join(seg_list)))
    seg_list = jieba.cut(sent, cut_all=False)
    log.info("精确模式:{}".format('/'.join(seg_list)))
    seg_list = jieba.cut(sent, )
    log.info("默认精确模式:{}".format('/'.join(seg_list)))
    seg_list = jieba.cut_for_search(sent)
    log.info("搜索引擎模式:{}".format('/'.join(seg_list)))


def test_news_topk():
    """
    测试新闻分词 后的topk 个主题词
    :return:
    """
    files = glob.glob(os.path.join(get_news_path(), '*.txt'))
    corpus = [get_content(file) for file in files]
    sample_inx = random.randint(0, len(corpus))
    split_words = [x for x in jieba.cut(corpus[sample_inx]) if x not in stop_words()]
    log.info('样本之一: %s' % corpus[sample_inx])
    log.info('样本分词后效果:%s ' % '/ '.join(split_words))
    log.info('样本的topK(10)词:%s' % str(get_TF(split_words)))


if __name__ == '__main__':
    # test_jieba()

    test_news_topk()

    pass