import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

'''
    PositionEncoding
    位置の情報を特徴に追加
'''
class PositionEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=2000):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (b, t, f)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

'''
    ASRTransformerDecoder
    attention重みを出力するためのカスタマイズしたデコーダー
'''
class ASRTransformerDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(ASRTransformerDecoder, self).__init__(decoder_layer, num_layers, norm)

    def _get_attention_weights(self):
        attention_weights=[]
        for mod in self.layers:
            attention_weights.append(mod._get_attention_weight().to('cpu').detach().numpy().copy())
        return attention_weights

'''
    ASRTransformerDecoderLayer
    Multi-head attention部でattention重みを出力して保存しておく
'''
class ASRTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True, norm_first=True):
        super(ASRTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, batch_first=batch_first, norm_first=norm_first)
        self.attention_weight=None

    def _mha_block(self, x, mem, attn_mask=None, key_padding_mask=None):
        x, self.attention_weight = self.multihead_attn(x, mem, mem,
                                                       attn_mask=attn_mask,
                                                       key_padding_mask=key_padding_mask,
                                                       need_weights=True)
        return self.dropout2(x)

    def _get_attention_weight(self):
        return self.attention_weight
