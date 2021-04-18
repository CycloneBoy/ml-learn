#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : decorator_learn.py
# @Author: sl
# @Date  : 2021/4/18 -  上午11:55

"""
python 装饰器模式 闭包
"""
import time
from functools import wraps
from decorator import decorator

def hint(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print('{} is running'.format(func.__name__))
        return func(*args, **kwargs)

    return wrapper


@hint
def hello():
    print("Hello!")




@decorator
def hint2(func, *args, **kwargs):
    print('{} is running'.format(func.__name__))
    return func(*args, **kwargs)


@hint2
def hello2():
    print("Hello2!")


def hint3(coder):
  def wrapper(func):
    @wraps(func)
    def inner_wrapper(*args, **kwargs):
      print('{} is running'.format(func.__name__))
      print('Coder: {}'.format(coder))
      return func(*args, **kwargs)
    return inner_wrapper
  return wrapper


@hint3(coder="John")
def hello3():
  print("Hello3!")


# 装饰器增加缓存功能
def cache(instance):
    def wrapper(func):
        @wraps(func)
        def inner_wrapper(*args, **kwargs):
          # 构建key: key => func_name::args::kwargs
            joint_args = ','.join((str(x) for x in args))
            joint_kwargs = ','.join('{}={}'.format(k, v) for k, v in sorted(kwargs.items()))
            key = '{}::{}::{}'.format(func.__name__,joint_args, joint_kwargs)
            # 根据key获取结果。如果key已存在直接返回结果，不用重复计算。
            result = instance.get(key)
            if result is not None:
                return result
            # 如果结果不存在，重新计算，缓存。
            result = func(*args, **kwargs)
            instance.set(key, result)
            return result
        return inner_wrapper
    return wrapper


# 创建字典构造函数，用户缓存K/V键值对
class DictCache:
    def __init__(self):
        self.cache = dict()

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value

    def __str__(self):
        return str(self.cache)

    def __repr__(self):
        return repr(self.cache)


# 创建缓存对象
cache_instance = DictCache()


# Python语法糖调用装饰器
@cache(cache_instance)
def long_time_func(x):
    time.sleep(x)
    return x


#类的装饰器写法， 不带参数
class Hint(object):
  def __init__(self, func):
    self.func = func

  def __call__(self, *args, **kwargs):
    print('{} is running'.format(self.func.__name__))
    return self.func(*args, **kwargs)


#类的装饰器写法， 带参数
class Hint2(object):
  def __init__(self, coder=None):
    self.coder = coder

  def __call__(self, func):
    @wraps(func)
    def wrapper(*args, **kwargs):
      print('{} is running'.format(func.__name__))
      print('Coder: {}'.format(self.coder))
      return func(*args, **kwargs)   # 正式调用主要处理函数
    return wrapper


# @Hint()
def hello4():
  print("Hello4!")

@Hint2(coder="John")
def hello5():
  print("Hello5!")

if __name__ == '__main__':

    hello()
    print(hello.__name__)

    hello2()
    print(hello2.__name__)

    hello3()

    # 调用装饰过函数
    # long_time_func(3)

    Hint(hello4)()
    hello5()
    pass
