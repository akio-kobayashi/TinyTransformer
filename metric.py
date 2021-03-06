import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

'''
    ctc_loss
    マルチタスク学習のためのCTC損失関数
'''
class ctc_loss(nn.Module):
    def __init__(self):
        super(ctc_loss, self).__init__()
        self.ctc=nn.CTCLoss()

    def forward(self, outputs, labels, output_lengths, label_lengths):
        # PyTorchのctc損失では時間次元を先頭に配置する必要があることに注意する
        outputs = rearrange(outputs, 'b t f -> t b f')

        return  self.ctc(outputs.cuda(), labels.cuda(),
                        output_lengths.cuda(), label_lengths.cuda())

'''
    ce_loss
    Transformerのためのcross-entropy損失
'''
class ce_loss(nn.Module):
    def __init__(self):
        super(ce_loss, self).__init__()
        self.ce=nn.CrossEntropyLoss()

    def forward(self, y_prd, y_ref, y_prd_len, y_ref_len):
        loss = 0.
        for b in range(y_prd.shape[0]):
            prd=y_prd[b, :y_prd_len[b], :]
            ref=y_ref[b, :y_ref_len[b]]
            loss += self.ce(prd,ref)

        return torch.mean(loss)

'''
    IterMeter
    optimizerによる更新回数を記録
'''
class IterMeter(object):
    def __init__(self):
        self._val = 0

    def step(self):
        self._val += 1

    def get(self):
        return self._val

'''
    _levenshtein_distance
    2つの系列のLevenshtein距離（編集距離）を計算
'''
def _levenshtein_distance(ref, hyp):
    m = len(ref)
    n = len(hyp)

    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n :
        ref, hyp = hyp, ref
        m, n = n, m

    distance = np.zeros((2, n+1), dtype=np.int32)

    for j in range(0, n+1):
        distance[0][j] = j

    for i in range(1, m+1):
        prev_row_idx = (i-1)%2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n+1):
            if ref[i-1] == hyp[j-1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j-1]
            else:
                s_num = distance[prev_row_idx][j-1]+1
                i_num = distance[cur_row_idx][j-1]+1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)
    return distance[m%2][n]

'''
    cer
    文字誤り率（音素誤り率）を計算
'''
def cer(ref, hyp):
    edit_distance = _levenshtein_distance(ref, hyp)

    if len(ref) == 0:
        raise ValueError("Length of reference should be greater than 0")
    cer = float(edit_distance)/len(ref)
    return cer
