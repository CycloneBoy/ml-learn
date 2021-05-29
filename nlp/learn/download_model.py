#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : download_model.py
# @Author: sl
# @Date  : 2021/5/26 -  下午10:01
from transformers import AutoModel

if __name__ == '__main__':
    AutoModel.from_pretrained('hfl/chinese-electra-small-generator', mirror='tuna')
    pass
