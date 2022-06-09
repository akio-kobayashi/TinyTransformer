import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from preprocess import PreNet
from einops import rearrange
from asr_transformer import ASRTransformerDecoder, ASRTransformerDecoderLayer, PositionEncoding
from metric import ctc_loss, ce_loss

'''
    ASRModel
    音声認識モデル
    transformerに特徴量と系列の前処理，損失，デコーダーを実装
'''
class ASRModel(nn.Module):
    def __init__(self, config):
        super(ASRModel, self).__init__()

        self.dim_input=config['dim_input']
        self.dim_output=config['dim_output']
        self.dim_model=config['dim_model']
        self.dim_feedforward=config['dim_feedforward']
        self.num_heads=config['num_heads']
        self.num_encoder_layers=config['num_encoder_layers']
        self.num_decoder_layers=config['num_decoder_layers']

        # 位置エンコーディング
        self.enc_pe = PositionEncoding(self.dim_input, max_len=2000)
        self.dec_pe = PositionEncoding(self.dim_model, max_len=256)

        # 離散シンボルからベクトル化(embedding)
        self.dec_embed = nn.Embedding(self.dim_output, self.dim_model)

        # 単純な畳み込みニューラルネットワーク
        # 音響特徴量の前処理
        self.prenet = PreNet(self.dim_input, self.dim_model)

        # transformer
        # attention weightが必要なため，デコーダーをカスタマイズ
        custom_decoder_layer = ASRTransformerDecoderLayer(self.dim_model, self.num_heads, self.dim_feedforward, batch_first=True, norm_first=True)
        custom_decoder = ASRTransformerDecoder(custom_decoder_layer, self.num_decoder_layers, nn.LayerNorm(self.dim_model))
        self.transformer = nn.Transformer(d_model=self.dim_model,
                                          nhead=self.num_heads,
                                          num_encoder_layers=self.num_encoder_layers,
                                          num_decoder_layers=self.num_decoder_layers,
                                          custom_decoder=custom_decoder,
                                          batch_first=True,norm_first=True)

        # transformer出力から系列出力へ
        self.fc = nn.Linear(self.dim_model, self.dim_output)
        self.fc_ctc = nn.Linear(self.dim_model, self.dim_output)

        # transformerの損失（CrossEntropyLoss）
        self.loss = ce_loss()
        # マルチタスク用 CTC損失 (CTCLoss)
        self.ctc = ctc_loss()
        # transformerの損失の重み
        self.weight = config['weight']

    def forward(self, inputs, labels, input_lengths, label_lengths):

        labels_in = labels[:, 0:labels.shape[-1]-1]
        labels_out = labels[:, 1:labels.shape[-1]]

        label_lengths -= 1
        y=self.prenet(self.enc_pe(inputs))
        z=self.dec_pe(self.dec_embed(labels_in))
        source_mask, target_mask, source_padding_mask, target_padding_mask = self.generate_masks(y, z, input_lengths, label_lengths)
        memory = self.transformer.encoder(y.cuda(),
                                          mask=source_mask.cuda(),
                                          src_key_padding_mask=source_padding_mask.cuda())


        y = self.transformer.decoder(z.cuda(), memory, tgt_mask=target_mask.cuda(),
                                     memory_mask=None,
                                     tgt_key_padding_mask=target_padding_mask.cuda(),
                                     memory_key_padding_mask=source_padding_mask.cuda())
        y=self.fc(y)

        loss=self.loss(y, labels_out, label_lengths, label_lengths)
        y_ctc = (1. - self.weight ) * self.fc_ctc(memory)
        loss += self.weight * self.ctc(y_ctc, labels_out, input_lengths, label_lengths)

        return y, loss

    def generate_masks(self, src, tgt, src_len, tgt_len):
        B=src.shape[0]
        S=src.shape[1]
        src_mask=torch.zeros((S,S), dtype=bool)
        T=tgt.shape[1]
        tgt_mask=self.generate_square_subsequent_mask(T)

        src_padding_mask=torch.ones((B, S), dtype=bool)
        tgt_padding_mask=torch.ones((B, T), dtype=bool)
        for b in range(B):
            src_padding_mask[b, :src_len[b]]=False
            tgt_padding_mask[b, :tgt_len[b]]=False

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def generate_square_subsequent_mask(self, seq_len):
        mask = (torch.triu(torch.ones((seq_len, seq_len))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def greedy_decode(self, src, src_len, max_len):
        with torch.no_grad():
            src_padding_mask = torch.ones(1, src.shape[1], dtype=bool).cuda()
            src_padding_mask[:, :src_len]=False

            y = self.prenet(self.enc_pe(src.cuda()))

            # transformer エンコーダ
            # 入力音響特徴量系列はすべて使うのでエンコーダを１回だけ伝播させる
            memory = self.transformer.encoder(y.cuda(), src_key_padding_mask=src_padding_mask)

            # 文頭記号 <bos>=2 のみからなるTensorを用意
            # デコードの進捗にともなってTensor ysは拡張していく
            ys=torch.ones(np.array[[2]], dtype=torch.int).cuda()
            memory_mask=None

            #   系列長が不明であるため外部変数max_lenで上限を与える
            for i in range(max_len - 1):
                mask=self.generate_square_subsequent_mask(ys.shape[1]).cuda()
                z = self.dec_pe(self.dec_embed(ys))

                # transformerデコーダへ
                # memoryは入力のすべてのコンテキスト，zはデコード済みの系列
                z = self.transformer.decoder(z, memory, tgt_mask=mask, memory_mask=memory_mask)
                atts = self.transformer.decoder._get_attention_weights()

                # torch.argmaxにより，出力確率が最大となるインデックス（id）を取得
                z = F.log_softmax(self.fc(z), dim=-1)
                z = torch.argmax(z[:, -1, :]).reshape(1, 1)

                # 系列に連結して拡張する
                ys = torch.cat((ys, z), dim=1) #(1, T+1)

                if z == 3: # 3は<eos>のid
                    break

            torch.cuda.empty_cache()

        ys = ys.to('cpu').detach().numpy().copy().squeeze()
        return ys.tolist(), atts
