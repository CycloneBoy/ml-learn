#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : stock_trading_first.py
# @Author: sl
# @Date  : 2020/9/5 - 下午11:12

from pandas_datareader.data import DataReader

df = DataReader('FB.US','quandl','2020-01-01','2018-02-01')

