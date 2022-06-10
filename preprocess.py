import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

'''
    CausalConv1d
    Dilated causal convolution（因果畳み込み）
    必ずしもこのような畳み込みを使う必要はない
'''
def CausalConv1d(in_channels, out_channels, kernel_size, dilation=1, stride=1, **kwargs):
   pad = (kernel_size - 1) * dilation
   return nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                    padding=pad, dilation=dilation, **kwargs)

'''
    ConvBlock
    畳み込みブロックの定義
    CausalConv1dに活性化関数と正規化を追加
    ブロックの構成はサンプルである
'''
class ConvBlock(nn.Module):
    def __init__(self, in_channels=1, out_channels=32, stride=1, dropout=0.1):
        super(ConvBlock, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.norm=nn.LayerNorm(in_channels)
        self.active=nn.ReLU()
        self.conv=CausalConv1d(in_channels, out_channels, kernel_size=5, stride=stride)
        self.dp=nn.Dropout(dropout)

    def forward(self, x):
        len=x.shape[1]
        x = self.norm(x)
        x = rearrange(x, 'b t f -> b f t')
        x = self.conv(self.active(x))
        x = self.dp(x)
        x = x[:, :, :len]
        x = rearrange(x, 'b f t -> b t f')
        return x

'''
    PreNet
    音響特徴量の前処理
    畳み込みを用いて識別に有効な特徴を抽出する
'''
class PreNet(nn.Module):
    def __init__(self, d_input=80, d_model=512, dropout=0.1):
        super(PreNet, self).__init__()

        layers=[]
        layers.append(ConvBlock(in_channels=d_input, out_channels=512))
        layers.append(ConvBlock(in_channels=512, out_channels=512))
        layers.append(ConvBlock(in_channels=512, out_channels=d_model))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x
