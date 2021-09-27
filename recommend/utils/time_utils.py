#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : time_utils.py
# @Author: sl
# @Date  : 2021/9/26 - 下午1:15
import random
import time


class Time:

    @staticmethod
    def time_trans(time_data):
        time_stamp = time_data
        time_array = time.localtime(time_stamp)
        return time.strftime("%Y-%m-%d %H:%M:%S", time_array)

    @staticmethod
    def get_local_time(set_time=time.time()):
        return Time.time_trans(set_time)

    @staticmethod
    def get_random_time(begin=1, end=30):
        seed = random.randint(begin * 24 * 3600, end * 24 * 3600)
        create_time = Time.get_local_time(time.time() - seed)
        return create_time


if __name__ == '__main__':
    res = Time.get_random_time()
    print(res)
