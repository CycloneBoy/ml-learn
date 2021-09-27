#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : common_utils.py
# @Author: sl
# @Date  : 2021/9/26 - 下午12:56

def get_tags(tags):
    id2tag = {i: label for i, label in enumerate(tags)}
    tag2id = {label: i for i, label in enumerate(tags)}

    return tag2id, id2tag
