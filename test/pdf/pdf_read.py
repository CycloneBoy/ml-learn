#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : pdf_read.py
# @Author: sl
# @Date  : 2021/11/19 - 下午10:25

"""
提取PDF中的文字


"""
from io import StringIO

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

def parse_pdf(file_name='samples/simple1.pdf'):
    output_string = StringIO()
    with open(file_name, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    print(output_string.getvalue())


def parse_one():
    print(pdfminer.__version__)
    file_name = '/home/sl/文档/pdf/论文/nlp/ocr/TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models_2109.10282.pdf'
    text = extract_text(file_name)
    print(text)
    print("-" * 20)


if __name__ == '__main__':
    pass
    # parse_one()
    # output_string = StringIO()
    # with open(file_name, 'rb') as fin:
    #     extract_text_to_fp(fin, output_string, laparams=LAParams(),
    #                        output_type='html', codec=None)
    file_name = f"{SAMPLE_DIR}/samples/simple1.pdf"
    parse_pdf(file_name)


    for page_layout in extract_pages(file_name):
        for element in page_layout:
            print(element)

    print("-" * 50)
    for page_layout in extract_pages(file_name):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                print(element.get_text())

    print("-" * 20)
    for page_layout in extract_pages(file_name):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    for character in text_line:
                        if isinstance(character, LTChar):
                            print(character.fontname)
                            print(character.size)



    print("-" * 20)
    # Open a PDF document.
    fp = open(file_name, 'rb')
    parser = PDFParser(fp)
    document = PDFDocument(parser)

    # Get the outlines of the document.
    outlines = document.get_outlines()
    for (level, title, dest, a, se) in outlines:
        print(level, title)