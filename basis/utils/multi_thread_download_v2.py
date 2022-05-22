#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ml-learn
# @File  : multi_thread_download_v2.py
# @Author: sl
# @Date  : 2022/5/21 - 下午10:09

import os
import queue
import random
import threading
import time
import urllib.request
from abc import ABC

from util.logger_utils import logger
from util.v2.file_utils import FileUtils

"""
多线程下载器
"""
queue_lock = threading.Lock()


# work_queue = queue.Queue()


class DownloadThread(threading.Thread):
    def __init__(self, work_queue, save_dir, exit_flag=0, save_file_type=None, sleep_max_second=3):
        threading.Thread.__init__(self)
        self.work_queue = work_queue
        self.flag = exit_flag
        self.save_dir = save_dir
        self.save_file_type = save_file_type
        self.sleep_max_second = sleep_max_second
        self.start_time = time.time()
        self.end_time = time.time()

    def run(self):
        # log.info("开启线程：" + self.name)
        self.start_time = time.time()
        self.download_file(self.name)
        # log.info("退出线程：" + self.name)

    def get_save_file_name(self, data):
        if isinstance(data, dict) and "url" in data:
            file_url = data["url"]
        else:
            file_url = data
        sp_filename = file_url.split("/")

        if isinstance(data, dict) and "file" in data:
            file_name = data["file"]
        elif self.save_file_type == "audio":
            file_name = f"{self.save_dir}/{sp_filename[len(sp_filename) - 2]}.mp3"
        else:
            file_name = self.save_dir + "/" + sp_filename[len(sp_filename) - 1]

        return file_url, file_name

    def download_file(self, thread_name):
        while not self.flag:
            # while not exitFlag:
            queue_lock.acquire()
            if not self.work_queue.empty():
                data = self.work_queue.get()
                queue_lock.release()

                file_url, file_name = self.get_save_file_name(data)
                try:
                    logger.info("开始下载文件: {} -> {}".format(file_url, file_name))
                    FileUtils.check_file_exists(filename=file_name)
                    urllib.request.urlretrieve(url=file_url, filename=file_name)
                except Exception as e:
                    print(e)
                    logger.info("下载文件出错: {} -> {}".format(data, file_name))
                    logger.error(e)

                self.end_time = time.time()
                usetime = self.end_time - self.start_time
                logger.info("线程 %s 下载完成一个文件,耗时：%s： %s" % (thread_name, usetime, data))
                self.sleep()
            else:
                queue_lock.release()

    def stop(self):
        self.flag = 1

    def sleep(self):
        sleep_time = random.random() * self.sleep_max_second
        logger.info(f"sleep_time:{sleep_time}")
        time.sleep(sleep_time)


class MultiTheadDownloader(ABC):
    def __init__(self, file_url_list, save_dir, thread_size=10, sleep_time=10, save_file_type=None, sleep_max_second=3):
        self.file_url_list = file_url_list
        self.save_dir = save_dir
        self.thread_size = thread_size
        self.sleep_time = sleep_time
        self.save_file_type = save_file_type
        self.sleep_max_second = sleep_max_second
        self.work_queue = queue.Queue()
        self.threads = []
        self.start_time = 0.0

        self.init_dir()

    def init_dir(self):
        if os.path.exists(self.save_dir):
            logger.info("dir %s is existed!" % self.save_dir)
        else:
            os.mkdir(self.save_dir)

    def start(self):
        self.start_time = time.time()
        logger.info("开始下载：{}".format(self.start_time))
        # 创建新线程
        for i in range(1, self.thread_size + 1):
            thread = DownloadThread(work_queue=self.work_queue, save_dir=self.save_dir, exit_flag=0,
                                    save_file_type=self.save_file_type, sleep_max_second=self.sleep_max_second)
            thread.start()
            self.threads.append(thread)

        # 填充队列
        queue_lock.acquire()
        for item in self.file_url_list:
            self.work_queue.put(item)
        queue_lock.release()

        # 等待队列清空
        while not self.work_queue.empty():
            pass

        logger.info(f"队列已经消费完毕!")
        return

    def stop_treads(self):
        for t in self.threads:
            t.stop()
        # 等待所有线程完成
        for t in self.threads:
            t.join()
        usetime = time.time() - self.start_time
        logger.info(f"下载总数:{len(self.file_url_list)} ，总共耗时：{usetime} 秒")
        logger.info("下载完成所有图片，退出主线程")
