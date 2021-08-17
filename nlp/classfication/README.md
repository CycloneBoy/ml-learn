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