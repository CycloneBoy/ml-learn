#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : text_sentiment_classification.py
# @Author: sl
# @Date  : 2020/10/11 - 下午4:50

import collections
import os
import random
import tarfile
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data

from  tqdm import tqdm
import sys

from util.common_utils import get_glove
from util.constant import IMDB_DATA_DIR, DATA_CACHE_DIR

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_ROOT = DATA_CACHE_DIR


fname = os.path.join(DATA_ROOT, 'aclImdb_v1.tar.gz')
if not os.path.exists(os.path.join(DATA_ROOT,"aclImdb")):
    print("从压缩包解压")
    with tarfile.open(fname,'r') as f:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(f, DATA_ROOT)



train_data ,test_data = d2l.read_imdb('train'),d2l.read_imdb('test')
print("train_data size:",len(train_data))
print("test_data size:",len(test_data))

vocab = d2l.get_vocab_imdb(train_data)
print('words in vocab:', len(vocab))
# words in vocab: 46152




batch_size = 64
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

for X,y in train_iter:
    print('X',X.shape,'y',y.shape)
    break

print('#batches:', len(train_iter))



embed_size, num_hiddens, num_layers = 100, 100, 2
net = d2l.BiRNN(vocab, embed_size, num_hiddens, num_layers)

print(net)
#batches: 391
# BiRNN(
#   (embedding): Embedding(46152, 100)
#   (encoder): LSTM(100, 100, num_layers=2, bidirectional=True)
#   (decoder): Linear(in_features=400, out_features=2, bias=True)
# )

glove_vocab = get_glove()



net.embedding.weight.url.copy_(
    d2l.load_pretrained_embedding(vocab.itos,glove_vocab))
net.embedding.weight.requires_grad = False # 直接加载预训练好的, 所以不需要更新它

# There are 21202 oov words.

lr , num_epochs = 0.01,5
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,net.parameters()),lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter,test_iter,net,loss,optimizer,device,num_epochs)

# training on  cuda
# epoch 1, loss 0.6060, train acc 0.646, test acc 0.784, time 162.6 sec
# epoch 2, loss 0.2142, train acc 0.807, test acc 0.815, time 165.5 sec
# epoch 3, loss 0.1211, train acc 0.838, test acc 0.837, time 166.0 sec
# epoch 4, loss 0.0791, train acc 0.865, test acc 0.844, time 165.2 sec
# epoch 5, loss 0.0544, train acc 0.889, test acc 0.849, time 165.2 sec



res = d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great']) # positive
print(res)

res = d2l.predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad']) # negative
print(res)


if __name__ == '__main__':
    pass
