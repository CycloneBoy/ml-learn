#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_post.py
# @Author: sl
# @Date  : 2020/11/10 - 下午11:24
import json
import urllib.request
import urllib.parse

url="http://www.mafengwo.cn/wo/ajax_post.php&sAction=getArticle&iPage=2&iUid=78849820"


headers = {
    "Host": 'www.mafengwo.cn',
    "Connection": 'keep-alive',
    "Content-Length": '40',
    "Accept": 'application/json, text/javascript, */*; q=0.01',
    "X-Requested-With": 'XMLHttpRequest',
    "User-Agent": 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.135 Safari/537.36',
    "Content-Type": 'application/x-www-form-urlencoded; charset=UTF-8',
    "Origin": 'http"://www.mafengwo.cn',
    "Referer": 'http"://www.mafengwo.cn/u/78849820/note.html',
    "Accept-Encoding": 'gzip, deflate',
    "Accept-Language": 'zh-CN,zh;q=0.9,en;q=0.8',
    "Cookie": '_ga=GA1.2.1727842978.1571065957; mfw_uuid=5da49065-a885-f57c-cc67-19082de4d53a; uva=s%3A91%3A%22a%3A3%3A%7Bs%3A2%3A%22lt%22%3Bi%3A157,065959%3Bs%3A10%3A%22last_refer%22%3Bs%3A23%3A%22http%3A%2F%2Fwww.mafengwo.cn%2F%22%3Bs%3A5%3A%22rhost%22%3BN%3B%7D%22%3B; _r=google; _rp=a%3A2%3A%7Bs%3A1%3A%22p%22%3Bs%3A15%3A%22www.google.com%2F%22%3Bs%3A1%3A%22t%22%3Bi%3A1577846582%3B%7D; UM_distinctid=1721c2218702ce-0c046e425811d9-14291003-1fa400-1721c2218718a7; c=QDGXrjBy-1592043336447-275acfdf549fb-103557; login=mafengwo; __jsluid_h=fb3186264d113f568632612441a8991e; __mfwurd=a%3A3%3A%7Bs%3A6%3A%22f_time%22%3Bi%3A1603637352%3Bs%3A9%3A%22f_rdomain%22%3Bs%3A0%3A%22%22%3Bs%3A6%3A%22f_host%22%3Bs%3A3%3A%22www%22%3B%7D; __mfwuuid=5da49065-a885-f57c-cc67-19082de4d53a; oad_n=a%3A3%3A%7Bs%3A3%3A%22oid%22%3Bi%3A1029%3Bs%3A2%3A%22dm%22%3Bs%3A15%3A%22www.mafengwo.cn%22%3Bs%3A2%3A%22ft%22%3Bs%3A19%3A%222020-11-07+23%3A26%3A13%22%3B%7D; __mfwc=direct; __omc_chl=; __omc_r=; _xid=twXS4uUk76XbQwKwzLaIALs4bDcTCPT9Jxf919gyGOvwvYYk4xWEv9nQYVRXuoNEl9eKgSNpIinkdc26JczUqg%3D%3D; mafengwo=4a728c79e5eb6dc4c0558c3d0287e521_33736599_5fa6bfc0e3b115.97200348_5fa6bfc0e3b179.13069985; _fmdata=OVA5UCNq25LOT4vKqvd%2BH6rlaM%2By3PTY%2BAeCppc3xa1XG1bmsgHbvngrcEUAWL4KodcdVnimkdc6NCCTjIRC8InVqMmYlB5rs5BtOLJRR4I%3D; __jsl_clearance=1605020374.363|0|KV%2FQU9uENfBa%2BxG4fWo5BSFzUXM%3D; PHPSESSID=06e9e6g2ss4kfb43lpn1sbno70; mfw_uid=33736599; __mfwa=1571065957785.96506.22.1604763013146.1605020385707; __mfwlv=1605020385; __mfwvn=18; Hm_lvt_8288b2ed37e5bc9b4c9f7008798d2de0=1603637352,1604763013,1605020386; CNZZDATA30065558=cnzz_eid%3D1616349978-1571062256-http%253A%252F%252Fwww.mafengwo.cn%252F%26ntime%3D1605018693; bottom_ad_status=0; uol_throttle=33736599; __mfwb=9eff42cf86de.4.direct; __mfwlt=1605021050; Hm_lpvt_8288b2ed37e5bc9b4c9f7008798d2de0=1605021050'
  }


req = urllib.request.Request(url=url,  headers=headers, method='POST')
response = urllib.request.urlopen(req).read()
print(response)

