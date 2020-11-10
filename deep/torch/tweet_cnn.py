#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : tweet_cnn.py
# @Author: sl
# @Date  : 2020/10/18 - 下午4:33

import os
import random
import re
import sys

import pandas as pd
import torch
import torch.utils.data as Data
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch import nn

from util.common_utils import get_glove, save_model
from util.constant import WORK_DIR, MODEL_NLP_DIR

sys.path.append("..")
import deep.torch.d2lzh_pytorch as d2l

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pd.set_option('display.max_colwidth', -1)

data_dir = "{}/data/test/tweet".format(WORK_DIR)

# nltk.download('stopwords')

stop = stopwords.words('english')
print(stop)


def print_text(data):
    print(data.head())


train_data = pd.read_csv("{}/{}".format(data_dir, 'train.csv'))
print_text(train_data)

test_data_eval = pd.read_csv("{}/{}".format(data_dir, 'test.csv'))
test_data_eval.head()

sample_submission = pd.read_csv("{}/{}".format(data_dir, 'sample_submission.csv'))
print_text(sample_submission)

train_data = train_data.drop(['keyword', 'location', 'id'], axis=1)
print_text(train_data)


# 文本常常包含许多特殊字符，这些字符对于机器学习算法来说不一定有意义。因此，我要采取的第一步是删除这些
def clean_text(df, text_field):
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


data_clean = clean_text(train_data, "text")
print_text(data_clean)

data_clean['text'] = data_clean['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print_text(data_clean)

X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)
print_text(X_train)
print_text(y_train)


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def read_tweet(data):
    result = []
    data_train = data['text']
    label_train = data['target']
    for line, label in zip(data_train, label_train):
        result.append([line, label])
    random.shuffle(result)
    return result


def read_tweet(X_train, y_train):
    result = []
    data_train = X_train
    label_train = y_train
    for line, label in zip(data_train, label_train):
        result.append([line, label])
    random.shuffle(result)
    return result


X_train, X_test, y_train, y_test = train_test_split(data_clean['text'], data_clean['target'], random_state=0)

train_data, test_data = read_tweet(X_train, y_train), read_tweet(X_test, y_test)
print("train_data size:", len(train_data))
print("test_data size:", len(test_data))
print(train_data)

vocab = d2l.get_vocab_imdb(train_data)
print('words in vocab:', len(vocab))
# words in vocab: 46152

batch_size = 64
train_set = Data.TensorDataset(*d2l.preprocess_imdb(train_data, vocab))
test_set = Data.TensorDataset(*d2l.preprocess_imdb(test_data, vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = Data.DataLoader(test_set, batch_size)

embed_size, kernel_sizes, nums_channels = 100, [1, 2, 3], [100, 100, 100]
net = d2l.TextCNN(vocab, embed_size, kernel_sizes, nums_channels)
print(net)

glove_vocab = get_glove()

net.embedding.weight.response.copy_(
    d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.constant_embedding.weight.response.copy_(
    d2l.load_pretrained_embedding(vocab.itos, glove_vocab))
net.constant_embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它

lr, num_epochs = 0.001, 5
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
loss = nn.CrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, optimizer, device, num_epochs)

# 保存模型
save_model(net, "tweet_cnn_model_1", MODEL_NLP_DIR)

# 重置FC 全连接层，再次进行迁移训练
num_fcs = net.decoder = nn.Linear(sum(nums_channels), 2)

train_news_data = pd.read_csv("{}/{}".format(data_dir, 'craw_new.csv'))
data_clean_news = clean_text(train_news_data, "text")
data_clean_news['text'] = data_clean_news['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

X_train_news = data_clean_news['text']
y_train_news = train_news_data['target']

train_data_news = read_tweet(X_train_news, y_train_news)
train_set_news = Data.TensorDataset(*d2l.preprocess_imdb(train_data_news, vocab))
train_news_iter = Data.DataLoader(train_set_news, batch_size, shuffle=True)
# 迁移模型训练
d2l.train(train_news_iter, test_iter, net, loss, optimizer, device, num_epochs)

test_line = "Building the perfect tracklist to life leave the streets ablaze"
res = d2l.predict_sentiment(net, vocab, [word for word in test_line.split(' ')])  # positive
print(res)

test_line = "During the 1960s the oryx a symbol of the Arabian Peninsula were annihilated by hunters.http://t.co/yangEQBUQW http://t.co/jQ2eH5KGLt"
res = d2l.predict_sentiment(net, vocab, [word for word in test_line.split(' ')])  # negative
print(res)


# 预测类别
def predict_sentiment(net, vocab, data):
    """sentence是词语的列表"""
    result = []
    for sentence in data:
        device = list(net.parameters())[0].device

        split = sentence.split(' ')
        if len(split) < 3:
            print("长度太短：" + sentence)
            result.append(0)
            continue
        sentence = torch.tensor([vocab.stoi[word] for word in split], device=device)
        label = torch.argmax(net(sentence.view((1, -1))), dim=1).sum().cpu().item()
        result.append(label)
    return result


sample_submission = pd.read_csv("{}/{}".format(data_dir, 'sample_submission.csv'))

submission_test_clean = test_data_eval.copy()
submission_test_clean = clean_text(submission_test_clean, "text")
submission_test_clean['text'] = submission_test_clean['text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
submission_test_clean = submission_test_clean['text']
submission_test_clean.head()
print_text(submission_test_clean)

submission_test_pred = predict_sentiment(net, vocab, submission_test_clean)

id_col = test_data_eval['id']
submission_df_1 = pd.DataFrame({
    "id": id_col,
    "target": submission_test_pred})
submission_df_1.head()
print_text(submission_df_1)

submission_df_1.to_csv("{}/{}".format(data_dir, 'submission_cnn3.csv'), index=False)


def predict_police(model, vocab):
    police_filename = "{}/{}".format(data_dir, 'police_clean.csv')
    police_data = pd.read_csv(police_filename)
    police_data_id = police_data["IR_No"]
    police_data_line = police_data["Main_Narrative"]
    # police_data_line_pred = model.predict(police_data_line)

    police_data_line_pred = predict_sentiment(model, vocab, police_data_line)

    police_data_df_1 = pd.DataFrame({
        "IR_No": police_data_id,
        "target(1:bad)": police_data_line_pred})
    print_text(police_data_df_1)
    police_data_df_1.to_csv("{}/{}".format(data_dir, 'police_data_df_cnn.csv'), index=False)


predict_police(net, vocab)

if __name__ == '__main__':
    pass
