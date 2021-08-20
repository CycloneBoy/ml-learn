# 基于Huggingface Transforms 实现bert 的ner任务

## 参考

https://github.com/bqw18744018044/BertForNER
https://blog.csdn.net/bqw18744018044/article/details/118445411
https://github.com/CLUEbenchmark/CLUENER2020

## 运行效果

### BertForNER

#### bert-base-chinese 5 epoch

```text
Tracking run with wandb version 0.10.33
Syncing run ./models/checkpoints to Weights & Biases (Documentation).
Project page: https://wandb.ai/cycloneboy/huggingface
Run page: https://wandb.ai/cycloneboy/huggingface/runs/39qky4zc
Run data is saved locally in /kaggle/working/wandb/run-20210820_031427-39qky4zc

 [420/420 05:39, Epoch 5/5]
Epoch	Training Loss	Validation Loss	F1	Precision	Recall
1	No log	0.456695	0.365094	0.425527	0.388475
2	No log	0.268093	0.669840	0.751356	0.674655
3	No log	0.226305	0.777240	0.775150	0.785237
4	No log	0.208818	0.792412	0.791028	0.795498
5	No log	0.215559	0.791061	0.786265	0.798027
 [11/11 00:02]


```

#### hfl/chinese-roberta-wwm-ext 5 epoch

```text
 Tracking run with wandb version 0.10.33
Syncing run ./models/checkpoints to Weights & Biases (Documentation).
Project page: https://wandb.ai/cycloneboy/huggingface
Run page: https://wandb.ai/cycloneboy/huggingface/runs/tmhobtvg
Run data is saved locally in /kaggle/working/wandb/run-20210820_042346-tmhobtvg

 [420/420 05:39, Epoch 5/5]
Epoch	Training Loss	Validation Loss	F1	Precision	Recall
1	No log	0.532605	0.323144	0.427197	0.348407
2	No log	0.280360	0.645581	0.715887	0.651701
3	No log	0.228041	0.774022	0.772040	0.782330
4	No log	0.215676	0.785289	0.784174	0.788778
5	No log	0.218269	0.788501	0.781947	0.796779
 [11/11 00:07]

```

#### hfl/chinese-roberta-wwm-ext 10 epoch

```text
  [840/840 11:39, Epoch 10/10]
Epoch	Training Loss	Validation Loss	F1	Precision	Recall
1	No log	0.452647	0.364199	0.436083	0.390126
2	No log	0.256539	0.713568	0.768139	0.701887
3	No log	0.230775	0.774396	0.766954	0.786128
4	No log	0.211439	0.792071	0.784034	0.802170
5	No log	0.220745	0.786049	0.781213	0.793653
6	0.342100	0.238769	0.781475	0.780459	0.784691
7	0.342100	0.248653	0.783276	0.774141	0.793234
8	0.342100	0.254722	0.785502	0.777876	0.795465
9	0.342100	0.264056	0.781020	0.774723	0.789593
10	0.342100	0.263541	0.783215	0.774956	0.792627
 [11/11 00:06]

```

### BertCrfForNER

#### bert-base-chinese 5 epoch

```text
Tracking run with wandb version 0.10.33
Syncing run ./models/checkpoints to Weights & Biases (Documentation).
Project page: https://wandb.ai/cycloneboy/huggingface
Run page: https://wandb.ai/cycloneboy/huggingface/runs/3lc0lpt7
Run data is saved locally in /kaggle/working/wandb/run-20210820_075827-3lc0lpt7

 [420/420 06:38, Epoch 5/5]
Epoch	Training Loss	Validation Loss	F1	Precision	Recall
1	No log	16.307722	0.422056	0.505713	0.429193
2	No log	9.824501	0.741018	0.764472	0.737758
3	No log	8.277965	0.794068	0.790437	0.799692
4	No log	8.327230	0.790424	0.787120	0.795338
5	No log	8.466567	0.791608	0.784400	0.800516
 [11/11 00:06]

```

#### hfl/chinese-roberta-wwm-ext 10 epoch

```text
 [840/840 13:26, Epoch 10/10]
Epoch	Training Loss	Validation Loss	F1	Precision	Recall
1	No log	20.100042	0.338609	0.387905	0.366847
2	No log	10.712934	0.682409	0.756752	0.669999
3	No log	9.126211	0.773210	0.766365	0.785200
4	No log	8.285098	0.784619	0.775850	0.795339
5	No log	8.874200	0.783515	0.775710	0.794278
6	14.018200	9.190708	0.787118	0.781031	0.795538
7	14.018200	9.466532	0.787640	0.778339	0.798485
8	14.018200	9.859019	0.788056	0.777760	0.799795
9	14.018200	10.203421	0.785700	0.775204	0.798042
10	14.018200	10.268199	0.786377	0.777533	0.796540

```

