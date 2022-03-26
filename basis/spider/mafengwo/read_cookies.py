#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : read_cookies.py
# @Author: sl
# @Date  : 2022/3/4 - 下午10:27


import requests
import http.cookiejar as cookielib

mafengwoSession = requests.session()
mafengwoSession.cookies = cookielib.LWPCookieJar(filename="mafengwoCookies.txt")

userAgent = "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"
header = {
    # "origin": "https://passport.mafengwo.cn",
    "Referer": "https://passport.mafengwo.cn/",
    'User-Agent': userAgent,
}


def mafengwoLogin(account, password):
    # 马蜂窝模仿 登录
    print("开始模拟登录马蜂窝")

    postUrl = "https://passport.mafengwo.cn/login/"
    postData = {
        "passport": account,
        "password": password,
    }
    # responseRes = requests.post(postUrl, data = postData, headers = header)
    url = "http://www.mafengwo.cn/travel-scenic-spot/mafengwo/10122.html"
    responseRes = requests.get(url, headers=header)
    # 无论是否登录成功，状态码一般都是 statusCode = 200
    print(f"statusCode = {responseRes.status_code}")
    print(f"text = {responseRes.text}")
    mafengwoSession.cookies.save()


if __name__ == "__main__":
    # 从返回结果来看，有登录成功
    mafengwoLogin("13756567832", "000000001")
