#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : data.py
# @Author: sl
# @Date  : 2021/4/13 -  下午10:44


"""
加载数据集
"""

import os
from codecs import open

from util.file_utils import save_to_text, save_to_json
from util.logger_utils import get_log

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


def build_map(lists):
    """
    根据词列表构建标示ID
    :param lists:
    :return:
    """
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return  maps

def build_corpus(split,file_end='.char.bmes',make_vocab=True,data_dir='../../data/ResumeNER'):
    """
    取数据
    :param split:   'train','dev','test'
    :param make_vocab: 是否构造词典
    :param data_dir: 数据路径
    :return:
    """
    assert split in ['train','dev','test']

    word_lists =[]
    tag_lists=  []
    with open(os.path.join(data_dir,split+file_end),'r',encoding='utf-8') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word,tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []

        # 如果make_vocab为True，还需要返回word2id和tag2id
        if make_vocab:
            word2id = build_map(word_lists)
            tag2id = build_map(tag_lists)
            return word_lists,tag_lists,word2id,tag2id
        else:
            return word_lists,tag_lists



if __name__ == '__main__':
    word_lists,tag_lists,word2id,tag2id = build_corpus('dev')
    log.info("数据集长度:{} 标签长度:{}".format(len(word_lists),len(tag_lists)))
    for idx in range(0,10):
        log.info("{} - {}".format(word_lists[idx],tag_lists[idx]))

    log.info("{}".format("-"*10))
    for index,tag in enumerate(tag2id):
        log.info("{} - {}".format(index, tag))


    save_to_json('../../data/ResumeNER/tag2id.json',tag2id)
    save_to_json('../../data/ResumeNER/word2id.json',word2id)


    pass