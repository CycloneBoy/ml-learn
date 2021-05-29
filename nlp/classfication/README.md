# 文本分类


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