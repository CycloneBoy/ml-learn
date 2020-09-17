#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : log_test.py
# @Author: sl
# @Date  : 2020/9/17 - 下午9:43

import logging
import time
from logging import handlers


# logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#                     level=logging.DEBUG)
#
# logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
#                     filename='new.log',
#                     filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     # a是追加模式，默认如果不写的话，就是追加模式
#                     format=
#                     '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
#                     # 日志格式
#                     )


def test_log():
    logging.debug('debug 信息')
    logging.warning('只有这个会输出。。。')
    logging.info('info 信息')


def test_log_info():
    logging.debug('debug 信息')
    logging.info('info 信息')
    logging.warning('warning 信息')
    logging.error('error 信息')
    logging.critical('critial 信息')



class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] -[%(threadName)s(%(process)d)]- %(levelname)s: %(message)s'):
        filename_str = get_log_file_name(filename)
        self.logger = logging.getLogger(filename_str)
        format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(format_str)  # 设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename_str, when=when, backupCount=backCount,
                                               encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        # 实例化TimedRotatingFileHandler
        # interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)  # 设置文件里写入的格式
        self.logger.addHandler(sh)  # 把对象加到logger里
        self.logger.addHandler(th)


def get_time():
    return time.strftime("%Y-%m-%d", time.localtime())


def get_log_file_name(file_name):
    name_list = str(file_name).split(".")
    return "{}_{}.{}".format(name_list[0], get_time(), name_list[1])


def get_log(filename, level='info'):
    my_logger = Logger(filename, level=level)
    return my_logger.logger


if __name__ == '__main__':
    # test_log()
    #
    # test_log_info()

    log = Logger('all.log', level='debug')
    log.logger.debug('debug')
    log.logger.info('info')
    log.logger.warning('警告')
    log.logger.error('报错')
    log.logger.critical('严重')
    Logger('error.log', level='error').logger.error('error')

    print(get_log_file_name("error.log"))
    pass
