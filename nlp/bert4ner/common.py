#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : common.py
# @Author: sl
# @Date  : 2021/8/22 - 上午9:01

import json
import pickle
from pathlib import Path


def save_pickle(data, file_path):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(input_file):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f)
    return data


def json_to_text(file_path, data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


if __name__ == '__main__':
    # save_pickle(predict_result, "./predict_result.pkl")
    predict_result = load_pickle("./predict_result.pkl")

    print(predict_result)
    pass
