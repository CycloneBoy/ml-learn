#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_glove.py
# @Author: sl
# @Date  : 2020/10/18 - 上午11:49
import os

import torch
from torchtext import  data
import torchtext.vocab as vocab
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors

from util.constant import DATA_CACHE_DIR, GLOVE_DATA_DIR

res = vocab.pretrained_aliases.keys()

vector_dir = '{}/glove/glove.6B'.format(DATA_CACHE_DIR)

print(res)
res = [key for key in vocab.pretrained_aliases.keys() if 'glove' in key]

print(res)


def test_glove(catch_dir):

    # glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
    glove = vocab.GloVe(name='6B', dim=50, cache=catch_dir)
    print("一共包含%d个词" % len(glove.stoi))
    print(glove.stoi['beautiful'])
    print(glove.itos[3366])


test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

sentence_set =set(test_sentence)
word_to_idx = {word: i+1 for i,word in enumerate(sentence_set)}
word_to_idx['<unk>'] = 0

idx_to_word = {i+1:word for i,word in enumerate(sentence_set)}

idx_to_word[0]='<unk>'


print(word_to_idx)
print(idx_to_word)

TEXT = data.Field(sequential=True)

if not os.path.exists(vector_dir):
    os.mkdir(vector_dir)

grove6b = "/glove.6B.200d.txt".format(vector_dir)
glove = vocab.Vectors(name="glove.6B.200d.txt",cache=GLOVE_DATA_DIR)
# TEXT.build_vocab(train,vertors=vectors)

print("一共包含%d个词" % len(glove.stoi))

print(glove.stoi['beautiful'])
print(glove.itos[3366])


if __name__ == '__main__':
    # test_glove()
    pass

