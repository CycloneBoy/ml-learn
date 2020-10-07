#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_voice.py
# @Author: sl
# @Date  : 2020/10/6 - 下午1:28


import pyttsx3
# engine = pyttsx3.init()

import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

engine = pyttsx3.init()
engine.setProperty('voice', 'zh')

engine.say('君不见，黄河之水天上来，奔流到海不复回。')
engine.say('君不见，高堂明镜悲白发，朝如青丝暮成雪。')
#运行并且等待


#engine.setProperty('volume', volume-0.25) 不明显
engine.setProperty('volume', volume-0.75)
engine.say('I will always love you')
engine.runAndWait()
