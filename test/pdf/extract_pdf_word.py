#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_pdf_word.py
# @Author: sl
# @Date  : 2021/11/20 - 下午2:41

"""
读取pdf 论文中的所有英文单词
"""
import enchant

from test.pdf.eudic_api import EuApi
from util.file_utils import save_to_text, get_dir, read_to_text
from util.logger_utils import logger

from io import StringIO
import re
import string

import pdfminer
from pdfminer.high_level import extract_text, extract_text_to_fp, extract_pages
from pdfminer.layout import LAParams, LTTextContainer, LTChar

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

SAMPLE_DIR = f"/home/sl/workspace/python/github/ocr/pdfminer.six"

TEST_PDF = '/home/sl/文档/pdf/论文/nlp/拼写纠错/READ_PLOME: Pre-training with Misspelled Knowledge_2021.acl-long.233.pdf'
file_name = TEST_PDF


def get_all_words(file_name):
    """
    获取所有的文字
    :return:
    """
    text = extract_text(file_name)

    save_file = f"{get_dir(file_name)}/test.txt"
    logger.info(f"保存解析的PDF: {file_name} ->text: {save_file}")
    save_to_text(save_file, text)
    return text


def filter_english(words):
    """
    filter wrong word
    :param words:
    :return:
    """
    enchant_util = enchant.Dict("en_US")
    filter_result = list(filter(enchant_util.check, words))
    # for word in words:
    #     if enchant_util.check(word):
    #         filter_result.append(word)
    logger.info(f"rc words: {len(words)} ,filer :{len(words) - len(filter_result)} ,result: {len(filter_result)}")
    return filter_result


def extract_english_word(file_name):
    text = read_to_text(file_name)
    pat = '[a-z]+'
    patter = re.compile(pat)
    words = patter.findall(str.lower(text))
    # 排序
    print(words)
    # l = sorted(words)
    result_word = filter_english(words)
    print(len(result_word))
    result = list(filter(lambda x: len(x) > 3, result_word))
    print(len(result))

    word_dict = {}
    for word in result:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    sort_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    print(sort_words)
    print(len(sort_words))

    upload_words = [key for key in word_dict.keys()]

    euapi = EuApi()
    res = euapi.add_words(cid="132818750585486525", words=upload_words)
    print(res)


def ad_word_to_eu():
    """

    :return:
    """
    pass


if __name__ == '__main__':
    pass

    # get_all_words(file_name)

    save_file = f"{get_dir(file_name)}/test.txt"
    extract_english_word(save_file)
    # all_pages = extract_pages(file_name)
    # for page_layout in all_pages:
    #     for element in page_layout:
    #         print(element)
    #
    #         if isinstance(element, LTTextContainer):
    #             for text_line in element:
    #                 for character in text_line:
    #                     if isinstance(character, LTChar):
    #                         print(f"fontname: {character.fontname}")
    #                         print(f"size: {character.size}")
    #                         print(f"ncs.name: {character.ncs.name}")
    #                         print(f"text: {character.get_text()}")
