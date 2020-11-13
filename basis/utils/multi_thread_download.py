#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:CycloneBoy
# datetime:2019/4/14 22:59

#!/usr/bin/python3
import os
import queue
import threading
import time
import urllib.request

from basis.utils.log import Logger
log = Logger(filename="/home/sl/workspace/python/a2020/ml-learn/data/log/multithread.log",level='debug').getLogger()


setImageDir = "/home/sl/workspace/python/mafengwo/imagehome"
exitFlag = 0
queueLock = threading.Lock()
workQueue = queue.Queue()


class myThread (threading.Thread):
    def __init__(self, workQueue,imageDir="image",exitflag=0):
        threading.Thread.__init__(self)
        self.q = workQueue
        self.flag = exitflag
        self.imageDir= imageDir
        self.starttime = time.time()
        self.endtime = time.time()

    def run(self):
        # log.info("开启线程：" + self.name)
        self.starttime = time.time()
        self.download_file(self.name, self.q)
        # log.info("退出线程：" + self.name)

    def download_file(self,threadName, q):
        while not self.flag:
        # while not exitFlag:
            queueLock.acquire()
            if not workQueue.empty():
                data = q.get()
                queueLock.release()
                fileurl = data
                sp_filename = fileurl.split("/")
                filename =  self.imageDir + "/" + sp_filename[len(sp_filename) - 1]
                urllib.request.urlretrieve(data, filename)
                self.endtime = time.time()
                usetime = self.endtime - self.starttime
                log.info("线程 %s 下载完成一张图片,耗时：%s： %s" % (threadName,usetime, data))
            else:
                queueLock.release()
            # time.sleep(1)

    def stop(self):
        self.flag = 1


def multi_download(imageList,imageDir="image",threadSize = 10):
    starttime = time.time()
    log.info("开始下载：{}".format(starttime))

    imageDir = os.path.join(setImageDir, str(imageDir).replace(" ","").replace("/",""))
    if os.path.exists(imageDir):
        log.info("dir %s is existed!" % imageDir)
    else:
        os.mkdir(imageDir)

    threads = []

    # 创建新线程
    for i in range(1, threadSize + 1):
        thread = myThread(workQueue,imageDir=imageDir)
        thread.start()
        threads.append(thread)

    # 填充队列
    queueLock.acquire()
    for image in imageList:
        workQueue.put(image)
    queueLock.release()

    # 等待队列清空
    while not workQueue.empty():
        pass
    return threads,starttime


def stop_treads(threads,starttime=0.0,imageList=None):
    for t in threads:
        t.stop()
    # 等待所有线程完成
    for t in threads:
        t.join()
    usetime = time.time() - starttime
    log.info("下载总数:{} ，总共耗时：{} 秒".format(len(imageList),usetime))
    log.info("下载完成所有图片，退出主线程")


if __name__ == '__main__':
    imageList = [
        "http://p1-q.mafengwo.net/s13/M00/B4/83/wKgEaVyivsKAIuWEABKtU4t2Xao210.png",
        "http://p1-q.mafengwo.net/s13/M00/B4/85/wKgEaVyivsOAbZPHAA9bYdv7j5Q332.png",
        "http://p2-q.mafengwo.net/s13/M00/3F/83/wKgEaVyh0W-ACiAMAAjM-m6ee6Y093.png",
        "http://p1-q.mafengwo.net/s13/M00/3F/81/wKgEaVyh0W6Aedq1AAnF2goc_B4959.png",
        "http://n2-q.mafengwo.net/s13/M00/3F/80/wKgEaVyh0W2AU1UTAAkYG56R6GY348.png",
        "http://n1-q.mafengwo.net/s13/M00/38/87/wKgEaVyhzvGAesGeAAo_g2wOa5g78.jpeg",
        "http://n4-q.mafengwo.net/s13/M00/79/AE/wKgEaVyhe7eAYvFkAA0lzmJzfFk32.jpeg",
        "http://n4-q.mafengwo.net/s13/M00/BD/7F/wKgEaVyE19GAJ5OFAAPfOvGKUnk355.png",
        "http://n1-q.mafengwo.net/s13/M00/F8/CF/wKgEaVx42qKAO1P7AAZYwfgO2kw13.jpeg"
    ]

    threads,starttime = multi_download(imageList,imageDir="image3")
    exitFlag=1
    stop_treads(threads,starttime,imageList)

