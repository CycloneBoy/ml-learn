#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : run_class_main.py
# @Author: sl
# @Date  : 2021/6/26 -  下午11:05

import xgboost as xgb

xgb_train = xgb.DMatrix("./agaricus.txt.train")
xgb_test = xgb.DMatrix("./agaricus.txt.test")

params = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eta": "1.0",
    "gamma": "1.0",
    "min_child_weight": 1,
    "max_depth": 3
}


def main():
    num_round = 2
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
    model = xgb.train(params, xgb_train, num_round, watchlist)
    model.save_model('./0002.model')


def eval_data():
    bst = xgb.Booster()
    bst.load_model('./0002.model')
    pred = bst.predict(xgb_test)
    print(pred)

    dump_model = bst.dump_model("./dump.raw.txt")
    dump_model = bst.dump_model("./dump.nice.txt",'./featmap.txt')



if __name__ == '__main__':
    # main()
    eval_data()
