import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import torch
import torch.nn as nn

class SpeechDataset(torch.utils.data.Dataset):

    def read_stats(self, path):
        with h5py.File(path, 'r') as f:
            self.mean = f['mean'][()]
            self.std = f['std'][()]

    def expand_mat(self, mat):
        rows=self.reduction - mat.shape[0]%self.reduction
        expanded=np.append(mat, np.repeat(mat[-1, :].reshape(1, -1), rows, axis=0), axis=0)
        return expanded

    def compose_mat(self, mat):
        assert (mat.shape[0]%self.reduction == 0)
        return mat.reshape(-1, mat.shape[1]*self.reduction)

    def decompose_mat(self, mat):
        dim=mat.shape[-1]
        return mat.reshape(1, -1, dim//self.reduction)

    def __init__(self, config):
        super(SpeechDataset, self).__init__()

        self.reduction=config['reduction']
        self.vtype=config['vocab_type']

        self.h5fd = h5py.File(config['data'], 'r')
        self.keys = [ key for key in self.h5fd.keys() ]
        self.read_stats(config['stats'])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        _key = self.keys[idx]
        _input = self.h5fd[self.keys[idx]+'/data'][()]
        if _input.shape[0] % self.reduction != 0:
            _input = self.expand_mat(_input)
        _input -= self.mean
        _input /= self.std
        if self.reduction > 1:
            _input=self.compose_mat(_input)

        _label = self.h5fd[self.keys[idx]+'/'+self.vtype][()]

        return _input, _label, _key


'''
    data_processing
    データ生成の後処理
'''
def data_processing(data):

    _inputs, _input_lengths = [], []
    _labels, _label_lengths = [], []
    _keys = []

    for input, label, key in data:
        _label_lengths.append(len(label))
        _labels.append(torch.from_numpy(label.astype(np.int)).clone())
        _input_lengths.append(input.shape[0])
        _inputs.append(torch.from_numpy(input.astype(np.float32)).clone())
        _keys.append(key)

    _inputs = nn.utils.rnn.pad_sequence(_inputs, batch_first=True)
    _labels = nn.utils.rnn.pad_sequence(_labels, batch_first=True)

    return _inputs, _labels, _input_lengths, _label_lengths, _keys
