#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:CycloneBoy
# datetime:2019/3/25 23:44
import logging
from logging import handlers
import datetime


class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self, filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

    def getLogger(self):
        return self.logger


# 获取日志文件
def getLoggerFileName(filename):
        # #日志记录
        to_day = datetime.datetime.now()
        log_file_path = filename + '_{}_{}_{}_{}_{}_{}.log'.format(to_day.year, to_day.month, to_day.day,
                                                                   to_day.hour, to_day.minute
                                                                   , to_day.second)
        return log_file_path


def log_test():
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        level=logging.DEBUG)

    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename='new.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        # 日志格式
                        )

    logging.debug('debug 信息')
    logging.warning('只有这个会输出。。。')
    logging.info('info 信息')

    logging.debug('debug级别，最低级别，一般开发人员用来打印一些调试信息')
    logging.info('info级别，正常输出信息，一般用来打印一些正常的操作')
    logging.warning('waring级别，一般用来打印警信息')
    logging.error('error级别，一般用来打印一些错误信息')
    logging.critical('critical 级别，一般用来打印一些致命的错误信息,等级最高')


def log_class_test():
    log = Logger('../log/test.log',level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')

    Logger('../log/error.log', level='error').logger.error('error')


if __name__ == '__main__':
    # log_test()
    log_class_test()
