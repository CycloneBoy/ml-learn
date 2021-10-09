#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : img_list.py
# @Author: sl
# @Date  : 2021/10/9 - 下午9:04

"""
爬去一页数据
"""
import json
import os
import time

from basis.utils.random_user_agent import RandomUserAgentMiddleware
from util.constant import DATA_HTML_DIR, DATA_QUESTION_DIR
from util.file_utils import save_to_text, read_to_text, check_file_exists, save_to_json
from util.logger_utils import get_log
from util.nlp_utils import sent2features, extend_maps, process_data_for_lstmcrf
import requests
from urllib import parse

log = get_log("{}.log".format(str(os.path.split(__file__)[1]).replace(".py", '')))

INDEX_USER_URL = "https://i.instagram.com/api/v1/feed/user/{}/username/?count=12"
INDEX_IMAGE_URL = "https://i.instagram.com/api/v1/feed/user/{}/?count=12&max_id={}"


def build_header(url):
    headers = {
        "authority": "i.instagram.com",
        "sec-ch-ua": '"Google Chrome";v="93", " Not;A Brand";v="99", "Chromium";v="93"',
        "x-ig-www-claim": "hmac.AR0xwEoAdv-QoXMh9jzi8rZfnoH0-SHGBqApNyXLp7RhyQ2l",
        "sec-ch-ua-mobile": "?0",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
        "accept": "*/*",
        "x-asbd-id": "198387",
        "sec-ch-ua-platform": "Linux",
        "x-ig-app-id": "936619743392459",
        "origin": "https:/www.instagram.com",
        "sec-fetch-site": "same-site",
        "sec-fetch-mode": "cors",
        "sec-fetch-dest": "empty",
        "referer": "https:/www.instagram.com/",
        "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
        "cookie": 'ig_did=E9EC0473-1CD0-4C7C-AE33-E00E9CE7D839; mid=Xc6hVgAEAAEGjqTYGm7_rVdYqaSt; csrftoken=NGGjfuhYuvnxMrcIAezJ2SOBfUFow76V; sessionid=2287083670%3AqE8hwXyuIIel6v%3A0; ig_nrcb=1; ds_user_id=2287083670; fbm_124024574287414=base_domain=.instagram.com; shbid="4547\0542287083670\0541665314060:01f74e5e0a52569a7653fb161d4a55decca3254a0a969ec488586721b3465b69a01a9665"; shbts="1633778060\0542287083670\0541665314060:01f77e6dd172acb9300e0960b0b7d7cc95cd8ee03db24b30928020fc3bec4d48cbe99f41"; fbsr_124024574287414=nYd8T7sJrvLOXCiSNbPo1mVCCu1nkNk2O9mQ4aEyieA.eyJ1c2VyX2lkIjoiMTAwMDA3MDg5MjI4ODI4IiwiY29kZSI6IkFRQzIyT0pRVURsZW1wREVwVjhrNEp3bTBZQ3JCODk3U2ZhMUR4ZEVNOFFZR3pFVGttT3F0NkRIOHlfNVNYYXZKb2p0WUQ3U1JOV0VjM0k2Sm15UlhIQXg2MjItU2xjdmpIMGRsc2dRcjhmTVhaMzJSN2dHWTFOd0lOS1VxNm40bTN2RjViRE1Oajd1aWVoOXp2Wmlmd25uT3drV0dCMlBfNHlHLUFfa2RWdDA2U3BMUU1HZDU4V3RiaGptVXlpNkNhVUYtTUYyZ3EzYUFkTmEzeWJVYzJpRU5ScHNOQ1N0dEh0MGQ4Y2ljUVBfUThvN0NnTkNoSmtnYVVFbUtfdlc2Nkx0bk40VGhKRVlldVhPMEFqclBoWGtTUzc3MlBja1pKT203SFBtZm1iZnFLRmU4Nl8yMDBuTl9hZlBrTU13ZmJVMl9qS2cwb2Q2d1E3bGhqYTktcUZWIiwib2F1dGhfdG9rZW4iOiJFQUFCd3pMaXhuallCQUFJY1JGVWpEYTFXREdRWkFJQUlLZnVZaXA0dHpBazBKUEd3VDNaQUhKeVpDNzRObzRaQjlaQzlFNkNXR01hdVlqdGpBRFZmQ1JEUlR4VWl3QWNGWkNxbVlFczVveExkMFBEc1hsYVNBaEVGQkpRN3BjWkNwODV0S01ERGpTTnhFWkFvWEF0WFFaQlE0c2h2VENyM1VSSUNaQ0JGUG1KZmtnclFOT01BNktVMFQyRzlNd1lNTVNQTzhaRCIsImFsZ29yaXRobSI6IkhNQUMtU0hBMjU2IiwiaXNzdWVkX2F0IjoxNjMzNzg0NTgxfQ; rur="VLL\0542287083670\0541665320876:01f703aacddba45f13aacde6a25796165a96615527439271ba0dc68d0edeb209e623d9d2"'
    }

    return headers


def send_http_request(url, data=None, method='GET'):
    """发送http 请求"""
    log.info("开始发送请求：{} - {} : {}".format(method, url, data))

    proxies = {'http': 'http://127.0.0.1:12333', 'https': 'http://127.0.0.1:12333'}
    if method.upper() == 'GET':
        r = requests.get(url, proxies=proxies, headers=build_header(url)).content.decode("utf-8")
    elif method.upper() == 'POST':
        if data is not None:
            r = requests.post(url, proxies=proxies, data=data, headers=build_header(url)).content.decode("utf-8")
        else:
            r = requests.post(url, proxies=proxies, headers=build_header(url)).content.decode("utf-8")
    else:
        r = ""

    log.info("请求的返回结果：{}".format(r))
    return r


def get_user_index_page(username, next_image=1):
    """获取用户首页 的具体内容"""

    url = INDEX_USER_URL.format(username)
    r = send_http_request(url=url, method='GET')
    # time.sleep(0.1)
    return r


def get_user_image_page(userid, next_image=""):
    """获取用户 图片的具体内容"""

    url = INDEX_IMAGE_URL.format(userid, next_image)
    r = send_http_request(url=url, method='GET')
    # time.sleep(0.1)
    return r


def parse_result(res, save_name="data/index.json"):
    """解析结果"""
    res_json = json.loads(res)
    save_to_json(save_name, res_json)

    userid = res_json["user"]["pk"]
    next_image_id = res_json["next_max_id"] if "next_max_id" in res_json else ""

    return res_json, userid, next_image_id


def get_user_image_list(username="syc_joycechu_"):
    """获取用户的首页图片信息"""

    index_res = get_user_index_page(username)

    res_json, userid, next_image_id = parse_result(index_res, save_name="data/index.json")

    index = 0
    while len(str(next_image_id)) > 0:
        index += 1
        log.info(f"开始爬取: {next_image_id}, 第{index} 页")
        res = get_user_image_page(userid, next_image=next_image_id)

        save_name = f"data/{next_image_id}.json"
        res_json, userid, next_image_id = parse_result(res, save_name=save_name)
        time.sleep(0.3)
        log.info(f"结束爬取: {next_image_id}, 第{index} 页")


if __name__ == '__main__':
    # username = "syc_joycechu_"
    # userid = "348953197"
    # # res = get_user_index_page(username)
    # next_image = "2624130966889641523_348953197"
    # res = get_user_image_page(userid, next_image=next_image)
    #
    # save_name = f"data/{next_image}.json"
    # parse_result(res, save_name=save_name)
    # # res = send_http_request(url="https://www.google.com/", method='GET')
    # # print(res)

    get_user_image_list(username="syc_joycechu_")
    pass
