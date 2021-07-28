#!/user/bin/env python
# -*- coding: utf-8 -*-
# @File  : seq2seq_v1.py
# @Author: sl
# @Date  : 2021/7/28 -  下午11:27

import argparse
import numpy as np
import torch
import torch.nn as nn


# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps


def make_batch():
    input_batch, output_batch, target_batch = [], [], []

    for seq in seq_data:
        for i in range(2):
            seq[i] = seq[i] + 'P' * (n_step - len(seq[i]))

        input = [num_dic[n] for n in seq[0]]
        output = [num_dic[n] for n in ('S' + seq[1])]
        target = [num_dic[n] for n in (seq[1] + 'E')]

        input_batch.append(np.eye(n_class)[input])
        output_batch.append(np.eye(n_class)[output])
        target_batch.append(target)

    return torch.FloatTensor(input_batch), torch.FloatTensor(output_batch), torch.LongTensor(target_batch)


class Seq2SeqModel(nn.Module):
    def __init__(self):
        super(Seq2SeqModel, self).__init__()

        self.enc_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.dec_cell = nn.RNN(input_size=n_class, hidden_size=n_hidden, dropout=0.5)
        self.fc = nn.Linear(n_hidden, n_class)

    def forward(self, enc_input, enc_hidden, dec_input):
        # enc_input: [max_len(=n_step, time step), batch_size, n_class]
        enc_input = enc_input.transpose(0, 1)
        # dec_input: [max_len(=n_step, time step), batch_size, n_class]
        dec_input = dec_input.transpose(0, 1)

        # enc_states : [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        _, enc_states = self.enc_cell(enc_input, enc_hidden)

        # outputs : [max_len+1(=6), batch_size, num_directions(=1) * n_hidden(=128)]
        output, _ = self.dec_cell(dec_input, enc_states)

        # model : [max_len+1(=6), batch_size, n_class]
        model = self.fc(output)
        return model


if __name__ == '__main__':
    n_step = 5
    n_hidden = 128
    num_epochs = 5000

    char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
    num_dic = {n: i for i, n in enumerate(char_arr)}
    seq_data = [['man', 'women'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'],
                ['high', 'low']]

    n_class = len(num_dic)
    batch_size = len(seq_data)

    model = Seq2SeqModel()
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

    input_batch, output_batch, target_batch = make_batch()

    for epoch in range(num_epochs):
        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1,batch_size,n_hidden)
        # input_batch : [batch_size, max_len(=n_step, time step), n_class]
        # output_batch : [batch_size, max_len+1(=n_step, time step) (becase of 'S' or 'E'), n_class]
        # target_batch : [batch_size, max_len+1(=n_step, time step)], not one-hot

        output = model(input_batch,hidden,output_batch)
        # output : [max_len+1, batch_size, n_class]

        # [batch_size, max_len+1(=6), n_class]
        output = output.transpose(0,1)
        loss = 0
        for i in range(0,len(target_batch)):
            # output[i] : [max_len+1, n_class, target_batch[i] : max_len+1]
            loss += criterion(output[i],target_batch[i])

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()



    def translate(word,n_hidden=n_hidden):
        input_batch, output_batch, _ = make_batch([[word, 'P' * len(word)]], args)

        # make hidden shape [num_layers * num_directions, batch_size, n_hidden]
        hidden = torch.zeros(1, 1, n_hidden)
        output = model(input_batch, hidden, output_batch)
        # output : [max_len+1(=6), batch_size(=1), n_class]

        predict = output.data.max(2, keepdim=True)[1] # select n_class dimension
        decoded = [char_arr[i] for i in predict]
        end = decoded.index('E')
        translated = ''.join(decoded[:end])

        return translated.replace('P', '')


    print('test')
    print('man ->', translate('man'))
    print('mans ->', translate('mans'))
    print('king ->', translate('king'))
    print('black ->', translate('black'))
    print('upp ->', translate('upp'))



