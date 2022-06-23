import numpy as np
import sys, os, re, gzip, struct
import random
import h5py

'''
    Vocab
    音声認識用語彙の処理
    音素、文字、モーラを扱えるが演習では音素のみを扱う
'''
class Vocab:
    def __init__(self, config):
        super(Vocab, self).__init__()
        self.word2id={}
        self.id2word={}

        # CTCで<blk>=0とするためにアルファベットは1からスタート
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

    # 語彙サイズを返す
    def size(self):
        return self.size_vocab

    # id列から文字列へ変換する
    # 本演習では，データはすでにid列に変換済みである
    # デコード結果を視認する際に用いる
    def to_string(self, seq):
        return [ self.id2word[n] for n in seq ]

    def remove_syms(self, seq):
        ret_seq=[]
        for s in seq:
            print(s)
            #if s == 2 or s == 3:
            #    continue
            ret_seq.append(s)
        return ret_seq
