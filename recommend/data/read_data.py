#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_data.py
# @Author: sl
# @Date  : 2021/9/26 - 下午12:09
import logging
import os.path
import re

from recommend.data.ArticleDao import ArticleDao
from recommend.model.DataModel import ArticleModel
from recommend.utils.Json import Json
from recommend.utils.common_utils import get_tags
from recommend.utils.constants import DATA_THUCNEWS_DIR2, NEWS_SUB_DIR
from recommend.utils.db_utils import MySql
from recommend.utils.time_utils import Time
from util.file_utils import list_file, read_to_text

# 只保留中英文、数字和.的正则表达式
cop = re.compile(u'[^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\'!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘]')


def read_data(filepath):
    contents = []
    with open(filepath, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        # lines = cop.sub("", "\n".join(line))
        contents.append(lines)
    return contents


def read_news(file_type, file_name, tag2id):
    """读取新闻"""
    file_dir = os.path.join(DATA_THUCNEWS_DIR2, file_type)
    file_path = os.path.join(file_dir, file_name)
    res = read_to_text(file_path)
    content_list = res.split("\n")
    title = content_list[0]
    content = "\n".join(content_list[1:])

    file_id = file_name[:str(file_name).index(".txt")]
    create_time = Time.get_random_time()
    news = ArticleModel(file_id, title, str(content).replace("'", '"'), tag2id[file_type], create_time)

    return news


def load_news_to_mysql(num=1000, end_num=1000, batch_size=100):
    """加载数据到mysql"""
    dir_list = NEWS_SUB_DIR
    tag2id, id2tag = get_tags(NEWS_SUB_DIR)

    mysql_base = init_mysql()
    for dir in dir_list:
        if dir == "房产":
            continue
        dir_name = os.path.join(DATA_THUCNEWS_DIR2, dir)
        file_list = list_file(dir_name)
        logging.info(f"目录：{dir} -  总共数量：{len(file_list)}　保存数量：{end_num - num}")

        news_list = []
        for file_name in file_list[num:end_num]:
            res = read_news(dir, file_name, tag2id)
            if not filter_news(res):
                news_list.append(res)

        news_dao = ArticleDao(mysql_base)

        for index in range(int((end_num - num) / batch_size)):
            news_dao.insert_art_batch(news_list[index * batch_size:(index + 1) * batch_size])


def init_mysql():
    path = os.path.join('../properties', 'database.json')
    db = Json.read_json_file(path)
    mysql_base = MySql(db_name=db['name'], user=db['user'], password=db['pass'],
                       host=db['host'], charset=db['charset'])
    return mysql_base


def filter_news(news: ArticleModel):
    """过滤非法字符信息 """
    # TODO：调整为正则过滤
    str_list = ["珠宝，得手后逃", "任务，直到今", "：王菲承载", "当今香港时尚界鼎鼎大名的Eason嫂"]
    for data in str_list:
        if str(news.content).find(data) > -1:
            return True
    if len(news.content) > 1000:
        return True
    return False


if __name__ == '__main__':
    load_news_to_mysql(num=2000, end_num=4000, batch_size=100)

    # line = "14å¹"
    # res = line.encode('utf-8')
    # print(res)
    # line = "号将在非洲之角海域执行保障船只航行安全的任务124SFVasv||,,,:::::,,,,,,"
    # res = cop.sub("", line)
    # print(res)
