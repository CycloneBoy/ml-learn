#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : ltp_test.py
# @Author: sl
# @Date  : 2020/9/22 - 下午10:36

from ltp import LTP
from transformers import BertModel, BertTokenizer

from util.logger_utils import get_log

log = get_log("{}.log".format("ltp_test"))

WORK_PATH = "/home/sl/workspace/python/a2020/ml-learn"
log.info("{}".format(WORK_PATH))


def test_ltp():
    ltp = LTP()
    # 分句
    sents = ltp.sent_split(["他叫汤姆去拿外衣。", "汤姆生病了。他去了医院。"])
    log.info("{}".format(sents))
    # 分词
    segment, _ = ltp.seg(["他叫汤姆去拿外衣。"])
    log.info("{}".format(segment))
    # 词性标注
    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    pos = ltp.pos(hidden)
    log.info("{}".format(seg))
    log.info("{}".format(pos))
    # 命名实体识别
    eg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    ner = ltp.ner(hidden)
    tag, start, end = ner[0][0]
    log.info("{} : {}".format(tag, "".join(seg[0][start:end + 1])))
    # 语义角色标注
    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    srl = ltp.srl(hidden)
    log.info("{}".format(srl))
    # 依存句法分析
    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    dep = ltp.dep(hidden)
    log.info("{}".format(dep))
    # 语义依存分析(树)
    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    sdp = ltp.sdp(hidden, graph=False)
    log.info("{}".format(sdp))
    # 语义依存分析(图)
    seg, hidden = ltp.seg(["他叫汤姆去拿外衣。"])
    sdp = ltp.sdp(hidden, graph=True)
    log.info("{}".format(sdp))


def test_read_file():
    with open('{}/data/txt/input_test.txt'.format(WORK_PATH), 'r') as f:
        lines = f.readlines()
        log.info("长度：{} ->{}".format(len(lines), lines))

        txt = f.read()
        txt_list = []
        txt_list.append(txt)
        ltp = LTP()
        ltp.init_dict(path="{}/data/txt/user_dict.txt".format(WORK_PATH), max_window=4)

        seg, hidden = ltp.seg(lines)
        ner = ltp.ner(hidden)
        log.info("ner:{} - len:{}".format(ner,len(ner)))
        log.info("seg:{} - len:{}".format(seg,len(seg)))
        # log.info("hidden:{}".format(hidden))

        # tag, start, end = ner[0][0]
        # log.info("{} : {}".format(tag, "".join(seg[0][start:end + 1])))
        print_word(seg,ner)
        log.info("{}".format("--------------语义角色标注-------------"))
        # srl = ltp.srl(hidden, keep_empty=False)
        # log.info("srl:{} - len:{}".format(srl,len(srl)))
        # print_s(seg,srl)


# 命名实体识别
def print_word(seg, ner):
    len_ner = len(ner)
    for index in range(0, len_ner):
        log.info("{}".format("----------------------------"))
        
        log.info("ner-index:{} - {} ".format(index,ner[index]))
        log.info("seg-index:{} - {} ".format(index,seg[index]))

        len_tag = len(ner[index])
        for i in range(0,len_tag):
            tag, start, end = ner[index][i]
            log.info("{}:{} ->{} : {}".format(index,i,tag, "".join(seg[index][start:end + 1])))


def print_s(seg,srl):
    len_ner = len(srl)
    for index in range(0, len_ner):
        log.info("{}".format("----------------------------"))

        log.info("srl-index:{} - {} ".format(index, srl[index]))
        log.info("seg-index:{} - {} ".format(index, seg[index]))

        len_tag = len(srl[index])
        for i in range(0, len_tag):
            tag, words = srl[index][i]
            log.info("{}".format("-------------------"))

            # log.info("{}:{} ->{} : {}".format(index, i, tag, words))
            for j in range(0,len(words)):
                tag1, start, end = words[j]
                log.info("{}:{} ->{} : {}".format(index, j, tag1, "".join(seg[index][start:end + 1])))


def print_dic():
    ltp = LTP()
    ltp.init_dict(path="{}/data/txt/user_dict.txt".format(WORK_PATH), max_window=4)
    pass

if __name__ == '__main__':
    # test_ltp()
    test_read_file()
    pass
