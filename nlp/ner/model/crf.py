#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : crf.py
# @Author: sl
# @Date  : 2021/4/18 -  上午11:13


"""
CRF  模型
"""

from sklearn_crfsuite import CRF

import os
from util.logger_utils import get_log
from util.nlp_utils import sent2features
from util.time_utils import time_cost

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


class CRFModel(object):
    def __init__(self, algorithm='lbfgs',
                 c1=0.1,
                 c2=0.1,
                 max_iterations=100,
                 all_possible_transitions=False):
        self.model = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    @time_cost
    def train(self, sentences, tag_lists):
        features = [sent2features(s) for s in sentences]
        self.model.fit(features, tag_lists)

    @time_cost
    def test(self, sentences):
        features = [sent2features(s) for s in sentences]
        pred_tag_lists = self.model.predict(features)
        return pred_tag_lists
