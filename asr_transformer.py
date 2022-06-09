import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy

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
