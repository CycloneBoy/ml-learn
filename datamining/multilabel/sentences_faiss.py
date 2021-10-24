#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : sentences_faiss.py
# @Author: sl
# @Date  : 2021/10/24 - 上午10:56


from sentence_transformers import SentenceTransformer, util

from sentence_transformers import models

import pandas as pd
import numpy as np

from util.constant import BERT_BASE_CHINESE_TORCH, QA_QUESTION_DATA_DIR, QA_DATA_DIR
from util.file_utils import save_to_pickle, save_to_text

model_name = BERT_BASE_CHINESE_TORCH
device = "cpu"
save_embedding_name = f"{QA_DATA_DIR}/qa_question_embedding.pickle"
save_pair_name = f"{QA_DATA_DIR}/qa_question_pair.pickle"


def get_model():
    # 使用 BERT 作为 encoder
    word_embedding_model = models.Transformer(model_name)
    # 使用 mean pooling 获得句向量表示
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    return model


def demo_word_embedding():
    model = get_model()
    sentences = ['This framework generates embeddings for each input sentence',
                 'Sentences are passed as a list of string.',
                 'The quick brown fox jumps over the lazy dog.']
    sentence_embeddings = model.encode(sentences)
    for sentence, embedding in zip(sentences, sentence_embeddings):
        print("Sentence:", sentence)
        print("Embedding:", embedding)
        print("")


def read_data(filename=QA_QUESTION_DATA_DIR):
    df = pd.read_csv(filename)

    print(df.info())

    res = df['question']

    result = res.tolist()
    return result


if __name__ == '__main__':
    # demo_word_embedding()

    qa_list = read_data()
    print(qa_list[:10])

    model = get_model()

    # Single list of sentences
    sentences = ['The cat sits outside',
                 'A man is playing guitar',
                 'I love pasta',
                 'The new movie is awesome',
                 'The cat plays in the garden',
                 'A woman watches TV',
                 'The new movie is so great',
                 'Do you like pizza?']

    sentences = qa_list

    # Compute embeddings
    embeddings = model.encode(sentences, convert_to_tensor=True)

    save_to_pickle(embeddings, save_embedding_name)
    print(f"保存embedding: {save_embedding_name}")

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

    # Find the pairs with the highest cosine similarity scores
    pairs = []
    for i in range(len(cosine_scores) - 1):
        for j in range(i + 1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    save_to_pickle(pairs, save_pair_name)
    print(f"保存pairs: {pairs}")

    save_to_text(f"{save_pair_name}.txt",
                 "\n".join([f"{pair['index'][0]},{pair['index'][1]},{pair['score']}" for pair in pairs]))

    for pair in pairs[0:10]:
        i, j = pair['index']
        print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], pair['score']))
