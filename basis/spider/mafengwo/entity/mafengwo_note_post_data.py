#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : MafengwoNotePostData.py
# @Author: sl
# @Date  : 2019/7/9 - 下午10:31
import json


class MafengwoNotePostData(object):
    def __init__(self, mddid, page, sort=1):
        self.mddid = mddid
        self.pageid = "mdd_index"
        self.sort = sort
        self.cost = 0
        self.days = 0
        self.month = 0
        self.tagid = 0
        self.page = page

    def json(self):
        return json.dumps(self.__dict__)


if __name__ == '__main__':
    postdata = MafengwoNotePostData("111",0)
    print(postdata.json())