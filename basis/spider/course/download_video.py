#!/user/bin/env python
# -*- coding: utf-8 -*-
# @Project ml-learn
# @File  : download_video.py
# @Author: sl
# @Date  : 2022/5/21 - 上午11:24


import requests
from Crypto.Cipher import AES
import execjs
import json
import os
import multiprocessing

APP_ID = "appixgujdjj6027"
PRUDUCT_ID = 'p_5ef84e6ac5b04_zl7ToAc5'
RESOURCE_ID = 'v_60780589e4b09134c98897ff'

BASE_URL = 'https://{0}.h5.xiaoeknow.com'.format(APP_ID.lower())

COOKIES = {
    'dataUpJssdkCookie': '{"wxver":"","net":"","sid":""}',
    'shop_version_type': '4',
    'anony_token': 'b3c38c849b9b25be080f542f4c8994c2',
    'xenbyfpfUnhLsdkZbX': '0',
    'sensorsdata2015jssdkcross': '%7B%22%24device_id%22%3A%22180dc923920658-0c18dbebb327e8-1b2e180b-2073600-180dc923921228%22%7D',
    'ko_token': '9f507b7544dfdbce2cf490160d81fd53',
    'sa_jssdk_2015_appixgujdjj6027_h5_xiaoeknow_com': '%7B%22distinct_id%22%3A%22u_61654a89415ec_I2puTirlss%22%2C%22first_id%22%3A%22180dc923920658-0c18dbebb327e8-1b2e180b-2073600-180dc923921228%22%2C%22props%22%3A%7B%22%24latest_traffic_source_type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%2C%22%24latest_referrer%22%3A%22%22%7D%7D',
    '_dd_s': 'logs=1&id=119eb367-df5f-4936-b666-57d8717f1c5d&created=1653059290656&expire=1653060210924',
    'logintime': '1653059311'
}

HEADERS = {

    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
    'sec-ch-ua': '^\\^Google',
    'Accept': 'application/json, text/plain, */*',
    'sec-ch-ua-mobile': '?0',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36',
    'Content-Type': 'application/x-www-form-urlencoded',
    'Origin': BASE_URL,
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Referer': '',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}


def get_CourseList():
    '''
    获取该课程下的每节课的基本信息：app_id，resource_id，title等
    :return:返回课程列表,主要是获取其中的课程对应id信息
    '''

    # last_id不要设置，order_weight按课程总数设置起始与终止数
    data = {

        'bizData[resource_id]': RESOURCE_ID,
        'bizData[product_id]': PRUDUCT_ID,
        'bizData[content_app_id]': '',
        'bizData[qy_app_id]': '',
        'bizData[page_num]': '40',
        'bizData[last_id]': '',
        'bizData[start_order_weight]': '0',
        'bizData[end_order_weight]': '41'
    }

    response = requests.post(BASE_URL + '/open_api/get_goods_catalogV3', headers=HEADERS,
                             cookies=COOKIES, data=data)
    lessons = response.json()['data']['content_info']

    for le in lessons:
        print(le)
    return lessons


def getBaseInfo(resource_id):
    '''
    请求单节课的详细信息
    :param resource_id: 课程id
    :return: 返回课程名与对应视频的m3u8,url
    '''

    # HEADERS[
    #     "Referer"] = "https://apppit6dcs05916.h5.xiaoeknow.com/v1/course/video/v_5e248ae02d88b_qcOkiiMV?type=2&pro_id=p_5d9eb71212cbe_Ckzdcjsp"

    postdata = {
        "type": "2",
        "product_id": PRUDUCT_ID,
        "resource_id": resource_id,
        "resource_type": 3,
        "app_id": APP_ID,
        "payment_type": ""}

    data = {

        'pay_info': json.dumps(postdata)}

    response = requests.post(BASE_URL + '/video/base_info', headers=HEADERS,
                             cookies=COOKIES, data=data)
    video_data = response.json()['data']['bizData']['data']
    title = video_data['title']
    encrypt_urls = video_data['video_urls']
    print(dectyptVideoInfo(encrypt_urls))

    # 6.15更新,改版后解密后字符串多了个符号，请求api里多了几个参数sign，t,us
    decrypt_urls = json.loads(dectyptVideoInfo(encrypt_urls)[:-1])
    print(decrypt_urls[0]['url'])
    # decrypt_urls = json.loads(dectyptVideoInfo(encrypt_urls))
    m3u8_url = decrypt_urls[0]['url']
    video_url = m3u8_url.split('drm')[0]
    sign = m3u8_url.split('m3u8?')[-1]
    # encrypt=decrypt_urls[0]['encrypt']

    video_url = video_url + "drm/v.f230.ts?" + sign

    return title, m3u8_url, video_url


def getVideoKey(m3u8_url):
    '''
    获取并解析解密视频内容的key值
    :param video_url: 视频url
    :return: 返回key值
    '''
    response = requests.get(m3u8_url)
    content = response.text

    uri_pos = content.find("URI")
    quotation_mark_pos = content.rfind('"')
    key_url = content[uri_pos:quotation_mark_pos].split('"')[1]

    key = requests.get(key_url).content

    return key


def dectyptVideoInfo(m3u8_response):
    '''
    对被加密的字符串进行解密，主要是替换一些特殊字符，之后再base64逆编码
    :param m3u8_response: 加密字符串
    :return: 解密后的字符串
    '''
    func = execjs.compile("""g = "3/a0_o6XeKn4zJqdc YMBDwfkPHI-pRig7hTWAjtyS=Em8FC9VbuvlGU+2rLxQ15.NOZs"
m = "H1G4w+vMuTqVQmE-0p3Ax79jDWZ_8hygIzeJPCi/=Lbk2FXdUNOc6fnloSaYrB5t.KRs "
y = "Ti/DQ5t8Udf4aoPqAVOpwnyC_Xxr IcRem71jLhKBguvGzs63=FJ-H0MZklb+WNSE9Y.2"


b = function (t) {
    // return t[y[27] + y[32] + m[17] + m[55] + y[12] + g[16] + m[34]](/[^A-Za-z0-9\+\/]/g, "")
    return t["replace"](/[^A-Za-z0-9\+\/]/g, "")
}

P = function (t) {
    // return b(t[y[27] + y[32] + m[17] + y[58] + g[2] + m[51] + y[32]](/[-_]/g, (function (t) {
    //     return t == y[52] ? y[60] : y[2]
    return b(t["replace"](/[-_]/g, (function (t) {
            return t == "-" ? "+" : "/"
        }
    )))
}

function dec_m3u8(t) {
    //原始代码
    // return t[y[1] + m[54] + y[9] + y[32] + y[26] + m[50] + y[10]](y[24] + g[50] + g[2]) > -1 ? (t = (t = (t = (t = (t = t[y[27] + m[34] + g[29] + y[58] + m[58] + y[30] + y[32]](m[27] + m[42] + g[2], ""))[y[27] + y[32] + m[17] + y[58] + m[58] + y[30] + m[34]](/@/g, y[35]))[m[60] + y[32] + y[19] + m[55] + y[12] + y[30] + y[32]](/#/g, m[44]))[m[60] + m[34] + g[29] + m[55] + g[2] + y[30] + g[8]](/\$/g, y[48]))[g[58] + g[8] + y[19] + g[53] + m[58] + y[30] + g[8]](/%/g, g[11]),
    //     I(P(t))) : t
    // //I就是base64解密函数

    return (t = (t = (t = (t = (t = t["replace"]("_ba", ""))["replace"](/@/g, "1"))["replace"](/#/g, "2"))["replace"](/\$/g, "3"))["replace"](/%/g, "4"),
        new Buffer(P(t), 'base64').toString())
}""")
    decryptedInfo = func.call("dec_m3u8", m3u8_response)
    return decryptedInfo


def downloadVideo(m3u8_url, video_url, save_path):
    '''
    下载视频
    :param video_url:
    :return:
    '''
    key = getVideoKey(m3u8_url)

    # video_url = video_url.split("drm")[0] + "drm/v.f230.ts"

    # 请求视频内容，并解密保存
    video = requests.get(video_url, headers=HEADERS, stream=True)
    cryptor = AES.new(key, AES.MODE_CBC, key)
    decrypted_video = cryptor.decrypt(video.content)

    with open(save_path, 'wb') as f:
        f.write(decrypted_video)


def main(lesson):
    # 视频保存路径
    save_space = "/home/sl/workspace/data/spider/course/video/"
    resource_id = lesson['resource_id']

    if resource_id[0] != "i":
        title, m3u8_url, video_url = getBaseInfo(resource_id)
        print("正在下载{0}......".format(title))
        video_full_path = save_space + title + ".mp4"
        if os.path.exists(video_full_path) == False:
            downloadVideo(m3u8_url, video_url, video_full_path)
            print("{0}下载完成".format(title))


if __name__ == "__main__":
    # 开多进程同时爬取多个视频
    pool = multiprocessing.Pool()
    lessons = get_CourseList()

    download_list = lessons[:1]
    pool.map(main, download_list)
    pool.close()
