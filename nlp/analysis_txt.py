#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : analysis_txt.py
# @Author: sl
# @Date  : 2020/9/22 - 下午9:55

import re
from datetime import datetime, timedelta

import jieba.posseg as psg
from dateutil.parser import parse

from util.logger_utils import get_log

log = get_log("{}.log".format("analysis_txt"))

UTIL_CN_NUM = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
               '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
               '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
               '5': 5, '6': 6, '7': 7, '8': 8, '9': 9
               }
UTIL_CN_UNIT = {'十': 10, '百': 100, '千': 1000, '万': 10000}


def check_time_valid(word):
    # 对提取的拼接日期串进行进一步处理，以进行有效性判断
    m = re.match("\d+$", word)
    if m:
        if len(word) <= 6:
            return None
    word1 = re.sub('[号|日]\d+$', '日', word)
    if word1 != word:
        return check_time_valid(word1)
    else:
        return word1


def cn2dig(src):
    if src == "":
        return None
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
            return int(datetime.datetime.today().year / 100) * 100 + int(m.group(0))
        else:
            return int(m.group(0))
    else:
        return None


def parse_datetime(msg):
    # 将每个提取到的文本日期串进行时间转换。
    print("parse_datetime开始处理：",msg)
    if msg is None or len(msg) == 0:
        return None
    try:
        msg = re.sub("年", " ", msg)  # parse不认识"年"字
        dt = parse(msg, yearfirst=True, fuzzy=True)
        # print(dt)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
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
            return target_date.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return None


def time_extract(text):
    time_res = []
    word = ''
    keyDate = {'今天': 0, '至今': 0, '明天': 1, '后天': 2}
    for k, v in psg.cut(text):
        if k in keyDate:
            if word != '':
                time_res.append(word)
            word = (datetime.today() + timedelta(days=keyDate.get(k, 0))).strftime('%Y年%m月%d日')
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
    print('time_res:{}'.format(time_res))
    result = list(filter(lambda x: x is not None, [check_time_valid(w) for w in time_res]))
    final_res = [parse_datetime(w) for w in result]
    return [x for x in final_res if x is not None]


def yanZhengRightData(string):
    m = re.match("(\d{4}|\d{2})-(\d{2}|\d{1})-(\d{2}|\d{1})", string)
    if m:
        return True
    else:
        return time_extract(string)


if __name__ == '__main__':
    log.info("{}".format("test_loader"))

    print(time_extract("从2016年3月5日至今"))
    print(time_extract("在20160824-20180529的全部交易。"))
    print(time_extract("2017.6.12-7.10交！"))
    print(time_extract("根据台湾军事迷记录，解放军军机在今天清晨7点52分、9点44分，分别进入台湾西南空域及北部空域，"
                       "高度分别在23000及9000米高度，台军方老调重弹，广播喊话要解放军军机注意，称“你已进入我空域，"
                       "影响我飞航安全，立即回转脱离。台媒引用长期记录的资深军事迷许先生分析，依照高度研判为运八反潜机，"
                       "同时在所谓“海峡中线”西侧，疑有1架解放军空警500型预警机，于高度6900米戒护，比对台湾防务部门发布的航迹图，解放军反潜机航迹均在相同位置，即为台湾海峡进入南海海盆入口，不排除解放军已在该处部署水下战力情况。对于解放军军机近日接连在台湾附近空域进行演习，台湾地区领导人对此回应称，大陆应保持克制。国防部新闻局副局长、国防部新闻发言人谭克非9月24日在例行记者会上表示，台湾是中国不可分割的一部分。解放军在台海地区组织实兵演练，展现的是捍卫国家主权和领土完整的决心和能力，针对的是外部势力干涉和极少数“台独”分裂分子及其分裂活动。台民进党当局置广大台湾同胞的安危福祉于不顾，不断挑动两岸对立对抗，进行“谋独”挑衅，危害台海和平稳定，这一图谋注定不会得逞。如果“台独”分裂势力胆敢以任何名义、任何方式把台湾从中国分裂出去，我们必将不惜一切代价，坚决予以挫败。"))
    print(yanZhengRightData("2017-12-21"))
    pass
