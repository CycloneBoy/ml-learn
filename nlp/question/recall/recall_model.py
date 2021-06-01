#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : recall_model.py
# @Author: sl
# @Date  : 2021/5/31 -  下午10:25

import time

from nlp.question.data.process_data import load_question
from nlp.question.recall.jieba_segment import Seg
from nlp.question.recall.sentence_similarity import SentenceSimilarity

if __name__ == '__main__':
    data_list = load_question(is_test=False)
    question_list = [qa.question for qa in data_list]
    answer_list = [qa.answer for qa in data_list]

    seg = Seg()
    ss = SentenceSimilarity(seg)
    ss.set_sentences(question_list)

    # ss.TfIdfModel()
    # ss.LsiModel()
    ss.LdaModel()

    while True:
        question = input("请输入问题(q退出): ")
        if question.lower() == 'q':
            break

        start_time = time.time()
        question_k = ss.similarity_k(question, 10)
        print("亲，我们给您找到的答案是： {}".format(answer_list[question_k[0][0]]))
        for idx, score in zip(*question_k):
            print("score： {:.4},same questions： {},answer:{} ".format(score, question_list[idx], answer_list[idx]))
        time2 = time.time()
        cost = time2 - start_time
        print('Time cost: {} s'.format(cost))

    pass
