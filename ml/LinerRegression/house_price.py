#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : 
# @Author: sl
# @Date  : 2021/2/9 - 15:52


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plot





if __name__ == '__main__':
    
    housing = pd.read_csv('kc_train.txt')
    target = pd.read_csv('kc_train2.txt')
    test_data = pd.read_csv('kc_test.txt')

    housing.info()

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(housing)
    scaler_housing = min_max_scaler.transform(housing)
    scaler_housing = pd.DataFrame(scaler_housing,columns=housing.columns)

    mm = MinMaxScaler()
    mm.fit(test_data)
    scaler_test = mm.transform(test_data)
    scaler_test = pd.DataFrame(scaler_test,columns=test_data.columns)

    LR_reg = LinearRegression()
    LR_reg.fit(scaler_housing,target)
    preds = LR_reg.predict(scaler_housing)
    mse = mean_squared_error(preds,target)

    plot.figure(figsize=0.7)
    num = 100
    x = np.array(1,num+1)
    plot.plot(x,target[:num],label="target")
    plot.plot(x,preds[:num],label="predict")
    plot.legend(loc="upper right")
    plot._show()


    result = LR_reg.predict(scaler_test)
    df_result = pd.DataFrame(result)
    df_result.to_csv("result.csv")







