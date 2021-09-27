#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : api_scheduler.py
# @Author: sl
# @Date  : 2021/9/27 - 下午9:58

"""

"""

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.executors.pool import ProcessPoolExecutor

def job_test():
    print("python")



if __name__ == '__main__':
    # 创建scheduler，多进程执行
    executors = {
        'default': ProcessPoolExecutor(3)
    }

    scheduler = BlockingScheduler(executors=executors)
    '''
     #该示例代码生成了一个BlockingScheduler调度器，使用了默认的默认的任务存储MemoryJobStore，
     以及默认的执行器ThreadPoolExecutor，并且最大线程数为10。
    '''
    scheduler.add_job(job_test, trigger='interval', seconds=5)
    '''
     #该示例中的定时任务采用固定时间间隔（interval）的方式，每隔5秒钟执行一次。
     #并且还为该任务设置了一个任务id
    '''
    scheduler.start()