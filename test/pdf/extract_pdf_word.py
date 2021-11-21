#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : extract_pdf_word.py
# @Author: sl
# @Date  : 2021/11/20 - 下午2:41

"""
读取pdf 论文中的所有英文单词
"""
import os.path
import time

import enchant

from test.pdf.eudic_api import EuApi
from util.file_utils import save_to_text, get_dir, read_to_text, list_file, get_file_name, save_to_json, \
    read_to_text_list
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

from util.time_utils import now_str

SAMPLE_DIR = f"/home/sl/workspace/python/github/ocr/pdfminer.six"

TEST_PDF = '/home/sl/文档/pdf/论文/nlp/拼写纠错/READ_PLOME: Pre-training with Misspelled Knowledge_2021.acl-long.233.pdf'
file_name = TEST_PDF
PDF_DIR = "/home/sl/文档/pdf/论文/nlp"

SAVE_TEXT_DIR = f"{PDF_DIR}/txt"


def get_all_words(file_name):
    """
    获取所有的文字
    :return:
    """
    text = extract_text(file_name)

    name = get_file_name(file_name)[:-4]
    save_file = f"{SAVE_TEXT_DIR}/{name}.txt"
    logger.info(f"保存解析的PDF: {file_name} ->text: {save_file}")
    save_to_text(save_file, text)
    return text, save_file


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
    logger.info(f"all words: {len(words)} ,filer :{len(words) - len(filter_result)} ,result: {len(filter_result)}")
    return filter_result


def filter_words(result_word, min_length=3):
    result = list(filter(lambda x: len(x) > min_length, result_word))
    logger.info(
        f"filter words length < {min_length} : {len(result_word)} ,filer :{len(result_word) - len(result)} ,result: {len(result)}")
    return result


def extract_english_word(file_name):
    """
    words to dict
    :param file_name:
    :return:
    """
    text = read_to_text(file_name)
    pat = '[a-z]+'
    pattern = re.compile(pat)
    words = pattern.findall(str.lower(text))
    # 排序
    print(words)
    # l = sorted(words)
    result_word = filter_english(words)
    print(len(result_word))
    result = list(filter(lambda x: len(x) > 3, result_word))
    print(len(result))

    word_dict = build_word_dict(result)
    sort_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    print(sort_words)
    print(len(sort_words))

    upload_words = [key for key in word_dict.keys()]

    euapi = EuApi()
    res = euapi.add_words(cid="132818750585486525", words=upload_words)
    print(res)


def build_word_dict(result):
    """
    uild word dict
    :param result:
    :return:
    """
    word_dict = {}
    for word in result:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

    return word_dict


def ad_word_to_eu():
    """

    :return:
    """
    pass


def extract_dict_pdf(pdf_dir):
    file_list = list_file(pdf_dir, endswith=".pdf")
    for index, file_name in enumerate(file_list):
        if not str(file_name).startswith("READ"):
            continue

        name = os.path.join(PDF_DIR, file_name)
        logger.info(f"begin process: {index} - {name}")
        text_name, save_file = get_all_words(name)
        logger.info(f"end process: {index} - {save_file}")


def extract_words_form_pdf(txt_dir):
    pattern = re.compile('[a-z]+')
    file_list = list_file(txt_dir, endswith=".txt")

    word_dict = {}
    for index, file_name in enumerate(file_list):
        name = os.path.join(txt_dir, file_name)

        text = read_to_text(name)
        words = pattern.findall(str.lower(text))
        result_word = filter_english(words)
        result = filter_words(result_word, min_length=3)
        word_dict[name] = result

    # save to text
    all_words = []
    for words in word_dict.values():
        all_words.extend(words)

    logger.info(f"total words : {len(all_words)}")
    save_file = f"{PDF_DIR}/all_words.txt"
    logger.info(f"保存all_words: {save_file}")
    save_to_text(save_file, "\n".join(all_words))

    word_dict = build_word_dict(all_words)
    sort_words = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    print(sort_words)
    save_file = f"{PDF_DIR}/all_words.json"
    save_to_json(save_file, sort_words)

    upload_words = [key for key in word_dict.keys()]
    save_file = f"{PDF_DIR}/upload_words.txt"
    logger.info(f"保存upload_words: {save_file} - {len(upload_words)}")
    save_to_text(save_file, "\n".join(upload_words))

    upload_to_eu(category_name="NLP_{}".format(now_str(format="%Y_%m_%d")))


def upload_to_eu(category_name="NLP2021_11_21"):
    euapi = EuApi()
    res = euapi.add_category(name=category_name)
    cid = res["id"]
    print(f" res: {res} - {cid}")

    save_file = f"{PDF_DIR}/upload_words.txt"
    upload_words = read_to_text_list(save_file)
    upload_words = [str(word).replace("\n", '') for word in upload_words]
    logger.info(f"保存upload_words: {save_file} - {len(upload_words)}")
    for index in range(0, len(upload_words), 500):
        # if index != 4500:
        #     continue
        upload_word = upload_words[index:index + 500]
        res = euapi.add_words(cid=cid, words=upload_word)
        logger.info(f"upload_word: {index} - {res}")
        time.sleep(0.5)


if __name__ == '__main__':
    pass

    # get_all_words(file_name)

    extract_dict_pdf(pdf_dir=PDF_DIR)
    extract_words_form_pdf(txt_dir=SAVE_TEXT_DIR)

    # save_file = f"{get_dir(file_name)}/test.txt"
    # extract_english_word(save_file)
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
