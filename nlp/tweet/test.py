#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: sl
# @Date  : 2020/10/12 - 下午10:28

import nltk

# nltk.download()

# nltk.download('stopwords')


import nltk
from nltk.book import *

import nltk

sen = 'hello, how are you?'
res = nltk.word_tokenize(sen)
print(res)


if __name__ == '__main__':
    pass
