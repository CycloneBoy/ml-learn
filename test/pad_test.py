#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : pad_test.py
# @Author: sl
# @Date  : 2021/4/18 -  下午1:35
import torch
from torch import tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

if __name__ == '__main__':
    
    seq = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
    lens = [2, 1, 3]
    packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
    print(packed)
    PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                   sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
    seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
    print(seq_unpacked)
    tensor([[1, 2, 0],
            [3, 0, 0],
            [4, 5, 6]])
    print(lens_unpacked)
    tensor([2, 1, 3])
    pass
