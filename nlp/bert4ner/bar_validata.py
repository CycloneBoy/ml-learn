#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : bar_validata.py
# @Author: sl
# @Date  : 2021/8/25 - 下午3:41
from random import random, randint
from time import sleep

from tqdm import tqdm, trange

from nlp.bert4ner.main import ProgressBar


def demo_bar():
    n_total = 100
    pbar = ProgressBar(n_total=n_total, desc='training')
    for step in range(n_total):
        pbar(step, {'loss': step})
        sleep(0.1)


def demo_bar1():
    for i in tqdm(range(1000000)):
        pass


def demo_bar2():
    text = ""
    for char in tqdm(["a", "b", "c", "d"]):
        sleep(0.25)
        text = text + char


def demo_bar3():
    for i in trange(100):
        sleep(0.01)


def demo_bar4():
    pbar = tqdm(["a", "b", "c", "d"])
    for char in pbar:
        sleep(0.25)
        pbar.set_description("Processing %s" % char)


def demo_bar5():
    with tqdm(total=100) as pbar:
        for i in range(10):
            sleep(0.1)
            pbar.update(10)


def demo_bar6():
    with trange(10) as t:
        for i in t:
            # Description will be displayed on the left
            t.set_description('GEN %i' % i)
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            t.set_postfix(loss=random(), gen=randint(1, 999), str='h',
                          lst=[1, 2])
            sleep(0.1)

    with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
              postfix=["Batch", dict(value=0)]) as t:
        for i in range(10):
            sleep(0.1)
            t.postfix[1]["value"] = i / 2
            t.update()


def demo_bar7():
    n_total = 100
    with trange(n_total) as t:
        for i in t:
            # Description will be displayed on the left
            t.set_description('[training] %i/%i' % (i, n_total))
            # Postfix will be displayed on the right,
            # formatted automatically based on argument's datatype
            t.set_postfix(loss=random(), gen=randint(1, 999), str='h',
                          lst=[1, 2])
            sleep(0.1)


if __name__ == '__main__':
    # demo_bar()
    demo_bar7()
