#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_squeeze.py
# @Author: sl
# @Date  : 2021/5/9 -  下午5:00

import torch

x = torch.Tensor([1, 2, 3, 4])  # torch.Tensor是默认的tensor类型（torch.FlaotTensor）的简称。

print('-' * 50)
print(x)  # tensor([1., 2., 3., 4.])
print(x.size())  # torch.Size([4])
print(x.dim())  # 1
print(x.numpy())  # [1. 2. 3. 4.]

print('-' * 50)
print(torch.unsqueeze(x, 0))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, 0).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, 0).dim())  # 2
print(torch.unsqueeze(x, 0).numpy())  # [[1. 2. 3. 4.]]

print('-' * 50)
print(torch.unsqueeze(x, 1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
print(torch.unsqueeze(x, 1).size())  # torch.Size([4, 1])
print(torch.unsqueeze(x, 1).dim())  # 2

print('-' * 50)
print(torch.unsqueeze(x, -1))
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
print(torch.unsqueeze(x, -1).size())  # torch.Size([4, 1])
print(torch.unsqueeze(x, -1).dim())  # 2

print('-' * 50)
print(torch.unsqueeze(x, -2))  # tensor([[1., 2., 3., 4.]])
print(torch.unsqueeze(x, -2).size())  # torch.Size([1, 4])
print(torch.unsqueeze(x, -2).dim())  # 2

# 边界测试
# 说明：A dim value within the range [-input.dim() - 1, input.dim() + 1) （左闭右开）can be used.
# print('-' * 50)
# print(torch.unsqueeze(x, -3))
# IndexError: Dimension out of range (expected to be in range of [-2, 1], but got -3)

# print('-' * 50)
# print(torch.unsqueeze(x, 2))
# IndexError: Dimension out of range (expected to be in range of [-2, 1], but got 2)

# 为何取值范围要如此设计呢？
# 原因：方便操作
# 0(-2)-行扩展
# 1(-1)-列扩展
# 正向：我们在0，1位置上扩展
# 逆向：我们在-2，-1位置上扩展
# 维度扩展：1维->2维，2维->3维，...，n维->n+1维
# 维度降低：n维->n-1维，n-1维->n-2维，...，2维->1维

# 以 1维->2维 为例，

# 从【正向】的角度思考：

# torch.Size([4])
# 最初的 tensor([1., 2., 3., 4.]) 是 1维，我们想让它扩展成 2维，那么，可以有两种扩展方式：

# 一种是：扩展成 1行4列 ，即 tensor([[1., 2., 3., 4.]])
# 针对第一种，扩展成 [1, 4]的形式，那么，在 dim=0 的位置上添加 1

# 另一种是：扩展成 4行1列，即
# tensor([[1.],
#         [2.],
#         [3.],
#         [4.]])
# 针对第二种，扩展成 [4, 1]的形式，那么，在dim=1的位置上添加 1

# 从【逆向】的角度思考：
# 原则：一般情况下， "-1" 是代表的是【最后一个元素】
# 在上述的原则下，
# 扩展成[1, 4]的形式，就变成了，在 dim=-2 的的位置上添加 1
# 扩展成[4, 1]的形式，就变成了，在 dim=-1 的的位置上添加 1

if __name__ == '__main__':

    pass