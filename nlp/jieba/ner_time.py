#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : ner_time.py
# @Author: sl
# @Date  : 2021/4/11 -  上午11:16

"""
命名实体识别
- 日期识别

"""

import os
import re
from datetime import datetime, timedelta

import jieba.posseg as psg
from dateutil.parser import parse

from util.logger_utils import get_log
import os

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))

UTIL_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
}

UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def check_length(msg):
    if msg is None or len(msg) == 0:
        return None
    return msg


def cn2dig(src):
    check_length(src)
    m = re.match("\d+", src)
    if m:
        return int(m.group(0))
    rsl = 0
    unit = 1
    for item in src[::-1]:
        if item in UTIL_CN_UNIT.keys():
            unit = UTIL_CN_UNIT[item]
        elif item in UTIL_CN_NUM.keys():
            num = UTIL_CN_NUM[item]
            rsl += num * unit
        else:
            return None
    if rsl < unit:
        rsl += unit
    return rsl


def year2dig(year):
    res = ''
    for item in year:
        if item in UTIL_CN_NUM.keys():
            res = res + str(UTIL_CN_NUM[item])
        else:
            res = res + item
    m = re.match("\d+", res)
    if m:
        if len(m.group(0)) == 2:
            return int(datetime.today().year / 100) * 100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


def parse_datatime(msg, formate='%Y-%m-%d %H:%M:%S'):
    check_length(msg)

    try:
        # dt = parse(msg, fuzzy=True)
        # return dt.strftime(formate)
    # except Exception as  e:
    #     log.info("无法正确解析日期,进行正则表达式解析:{}".format(e))
        m = re.match(
            r"([0-9零一二两三四五六七八九十]+年)?([0-9一二两三四五六七八九十]+月)?([0-9一二两三四五六七八九十]+[号日])?([上中下午晚早]+)?([0-9零一二两三四五六七八九十百]+[点:\.时])?([0-9零一二三四五六七八九十百]+分?)?([0-9零一二三四五六七八九十百]+秒)?",
            msg)
        if m.group(0) is not None:
            res = {
                "year": m.group(1),
                "month": m.group(2),
                "day": m.group(3),
                "hour": m.group(5) if m.group(5) is not None else '00',
                "minute": m.group(6) if m.group(6) is not None else '00',
                "second": m.group(7) if m.group(7) is not None else '00',
            }
            params = {}

            for name in res:
                if res[name] is not None and len(res[name]) != 0:
                    tmp = None
                    if name == 'year':
                        tmp = year2dig(res[name][:-1])
                    else:
                        tmp = cn2dig(res[name][:-1])
                    if tmp is not None:
                        params[name] = int(tmp)
            target_date = datetime.today().replace(**params)
            is_pm = m.group(4)
            if is_pm is not None:
                if is_pm == u'下午' or is_pm == u'晚上' or is_pm == '中午':
                    hour = target_date.time().hour
                    if hour < 12:
                        target_date = target_date.replace(hour=hour + 12)
            return target_date.strftime(formate)
        else:
            return None
    except Exception as  e:
        return None


def check_time_valid(word):
    m = re.match("\d+$", word)
    if m:
        if len(word) <= 6:
            return None
    word1 = re.sub('[号|日]\d+$', '日', word)
    if word1 != word:
        return check_time_valid(word1)
    else:
        return word1


def time_extract(text):
    time_res = []
    word = ''
    key_date = {'今天': 0, '明天': 1, '后天': 2}
    for k, v in psg.cut(text):
        if k in key_date:
            if word != '':
                time_res.append(word)
            word = (datetime.today() + timedelta(days=key_date.get(k, 0))).strftime('%Y年%m月%d日')
        elif word != '':
            if v in ['m', 't']:
                word = word + k
            else:
                time_res.append(word)
                word = ''
        elif v in ['m', 't']:
            word = k
    if word != '':
        time_res.append(word)
    result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    for w in result:
        log.info("{}".format(w))

    final_res = [parse_datatime(w) for w in result]
    return [x for x in final_res if x is not None]


if __name__ == '__main__':
    res = parse_datatime('2021年04月12日下午三点')
    print(res)

    text1 = '我要住到明天下午三点'
    print(text1, time_extract(text1), sep=':')

    text2 = '预定28号的房间'
    print(text2, time_extract(text2), sep=':')

    text3 = '我要从26号下午4点住到11月2号'
    print(text3, time_extract(text3), sep=':')

    text4 = '我要预订今天到30的房间'
    print(text4, time_extract(text4), sep=':')

    text5 = '今天30号呵呵'
    print(text5, time_extract(text5), sep=':')
