# 文本分类

## 实验效果

```text
效果
模型	acc	备注
TextCNN	91.22%	Kim 2014 经典的CNN文本分类
TextRNN	91.12%	BiLSTM
TextRNN_Att	90.90%	BiLSTM+Attention
TextRCNN	91.54%	BiLSTM+池化
FastText	92.23%	bow+bigram+trigram， 效果出奇的好
DPCNN	91.25%	深层金字塔CNN
Transformer	89.91%	效果较差
bert	94.83%	bert + fc
ERNIE	94.61%	比bert略差(说好的中文碾压bert呢)

模型	acc	备注
bert	94.83%	单纯的bert
ERNIE	94.61%	说好的中文碾压bert呢
bert_CNN	94.44%	bert + CNN
bert_RNN	94.57%	bert + RNN
bert_RCNN	94.51%	bert + RCNN
bert_DPCNN	94.47%	bert + DPCNN

```

### model

```text
使用Huggingface-Transformers
依托于Huggingface-Transformers 2.2.2，可轻松调用以上模型。

tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BertModel.from_pretrained("MODEL_NAME")
注意：本目录中的所有模型均使用BertTokenizer以及BertModel加载，请勿使用RobertaTokenizer/RobertaModel！

其中MODEL_NAME对应列表如下：

模型名	MODEL_NAME
RoBERTa-wwm-ext-large	hfl/chinese-roberta-wwm-ext-large
RoBERTa-wwm-ext	hfl/chinese-roberta-wwm-ext
BERT-wwm-ext	hfl/chinese-bert-wwm-ext
BERT-wwm	hfl/chinese-bert-wwm
RBT3	hfl/rbt3
RBTL3	hfl/rbtl3


[1] WWM = Whole Word Masking
[2] ext = extended data
[3] TPU Pod v3-32 (512G HBM)等价于4个TPU v3 (128G HBM)
[4] ~BERT表示继承谷歌原版中文BERT的属性

篇章级文本分类：THUCNews
篇章级文本分类任务我们选用了由清华大学自然语言处理实验室发布的新闻数据集THUCNews。 我们采用的是其中一个子集，需要将新闻分成10个类别中的一个。 评测指标为：Accuracy

模型	开发集	测试集
BERT	97.7 (97.4)	97.8 (97.6)
ERNIE	97.6 (97.3)	97.5 (97.3)
BERT-wwm	98.0 (97.6)	97.8 (97.6)
BERT-wwm-ext	97.7 (97.5)	97.7 (97.5)
RoBERTa-wwm-ext	98.3 (97.9)	97.7 (97.5)
RoBERTa-wwm-ext-large	98.3 (97.7)	97.8 (97.6)

```

## 运行命令行

```shell

# 执行命令 
/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
 --model TextCNN \
 --embedding pre_trained


tensorboard   --port 8123  --logdir=/home/sl/workspace/python/a2020/ml-learn/data/nlp/THUCNews/log/TextCNN/05-09_21.50



/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
 --model FastText \
 --embedding pre_trained
 
 
/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model TextRNN \
--embedding pre_trained
 
 
 /home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model TextRNN_Att \
--embedding pre_trained
 
 
  /home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model Transformer \
--embedding pre_trained


/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model bert 


/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model BertRCNN 


/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model BertCNN 
 
/home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model BertDPCNN 
 
 
 /home/sl/app/anaconda3/bin/python \
/home/sl/workspace/python/a2020/ml-learn/nlp/classfication/run.py \
--model electra 
 
 
 

```

## 此处存放ERNIE预训练模型：

pytorch_model.bin  
bert_config.json  
vocab.txt

## 下载地址：

http://image.nghuyong.top/ERNIE.zip  