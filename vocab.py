import numpy as np
import sys, os, re, gzip, struct
import random
import h5py

class Vocab:
    def __init__(self, config):
        super(Vocab, self).__init__()
        self.word2id={}
        self.id2word={}

        self.size_vocab=1
        self.vtype=config['vocab_type']
        if self.vtype != 'char' and self.vtype != 'phone' and self.vtype != 'mora' and self.vtype != 'label':
            raise ValueError('not a type %s' % self.vtype)
        with open(config['vocab'], 'r') as f:
            lines = f.readlines()
            for line in lines:
                tokens=line.strip().split()
                self.word2id[tokens[0]]=int(tokens[1])
                self.id2word[int(tokens[1])]=tokens[0]
                self.size_vocab+=1

    def size(self):
        return self.size_vocab

    def to_string(self, seq):
        return [ self.id2word[n] for n in seq ]
