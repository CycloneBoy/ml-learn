#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : time_utils.py
# @Author: sl
# @Date  : 2021/4/18 -  上午11:37

"""
时间工具类函数
"""
import time
import datetime
import timeit
from functools import wraps

from decorator import decorator
import os
from util.logger_utils import get_log

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))


class TimeUtils(object):
    std_format = '%Y-%m-%d %H:%M:%S'
    date_format = '%Y-%m-%d'

    now = datetime.datetime.now()
    now_str = time.strftime(std_format, time.localtime())
    now_date_str = time.strftime(date_format, time.localtime())


def now():
    return datetime.datetime.now()


def now_str(format="%Y-%m-%d %H:%M:%S"):
    return time.strftime(format, time.localtime())


def time_cost(func):
    """函数运行时间装饰器函数"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        # 执行函数
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        use_time = end - start
        if use_time * 1000 < 1000:
            use_time_str = "{:.3f} ms".format(use_time * 1000)
        else:
            use_time_str = "{:.3f} s".format(use_time)
        show_message = 'current Function [{}.{}] run time is {} '.format(func.__module__, func.__name__, use_time_str)
        print(show_message)
        log.info(show_message)
        return result

    return wrapper


# 类的装饰器写法， 带参数
class TimeCost(object):
    # def __init__(self):
    #     pass

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = timeit.default_timer()
            func(*args, **kwargs)
            end = timeit.default_timer()
            print('current Function [{}] run time is {:.3f} s'.format(func.__name__, end - start))

        return wrapper


@TimeCost()
def sleep2():
    time.sleep(2)


@time_cost
def sleep1():
    time.sleep(1)


if __name__ == '__main__':
    res = now()
    print(res)

    res = time_cost(now)
    print(res)

    sleep2()
    sleep1()
    pass
