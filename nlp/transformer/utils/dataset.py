#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : dataset.py
# @Author: sl
# @Date  : 2021/4/24 -  下午5:55


import torchtext.data as data

class ParallelDataset(data.Dataset):
    """
    定义一个自定义的数据集
    Defines a custom dataset for machine translation.
    """
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src),len(ex.trg))

    def __init__(self,src_examples,trg_examples,field,**kwargs):
        """Create a Translation Dataset given paths and fields.

        :param src_examples:
        :param trg_examples:
        :param field:
        :param kwargs:
        """
        if not isinstance(field[0],(tuple,list)):
            if trg_examples is None:
                fields = [('src',field[0])]
            else:
                fields = [('src',field[0]),('trg',field[1])]

        examples = []
        if trg_examples is None:
            for src_line in src_examples:
                examples.append(data.Example.fromlist([src_line],fields))
        else:
            for src_line,trg_line in zip(src_examples,trg_examples):
                examples.append(data.Example.fromlist([src_line,trg_line], fields))


        super(ParallelDataset,self).__init__(examples,fields,**kwargs)