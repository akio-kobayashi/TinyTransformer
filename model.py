import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from preprocess import PreNet
from einops import rearrange
from asr_transformer import ASRTransformerDecoder, ASRTransformerDecoderLayer

class ctc_loss(nn.Module):
    def __init__(self):
        super(ctc_loss, self).__init__()
        self.ctc=nn.CTCLoss()

    def forward(self, outputs, labels, output_lengths, label_lengths):
        outputs = rearrange(outputs, 'b t f -> t b f')

        return  self.ctc(outputs.cuda(), labels.cuda(),
                        output_lengths.cuda(), label_lengths.cuda())

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
        self.enc_pe = PositionEncoding(self.dim_input, max_len=2000)
        self.dec_pe = PositionEncoding(self.dim_model, max_len=256)
        self.dec_embed = nn.Embedding(self.dim_output, self.dim_model)
        self.prenet = PreNet(self.dim_input, self.dim_model)
        custom_decoder_layer = ASRTransformerDecoderLayer(self.dim_model, self.num_heads, self.dim_feedforward, batch_first=True, norm_first=True)
        custom_decoder = ASRTransformerDecoder(custom_decoder_layer, self.num_decoder_layers, nn.LayerNorm(self.dim_model))
        self.transformer = nn.Transformer(d_model=self.dim_model,
                                          nhead=self.num_heads,
                                          num_encoder_layers=self.num_encoder_layers,
                                          num_decoder_layers=self.num_decoder_layers,
                                          custom_decoder=custom_decoder,
                                          batch_first=True,norm_first=True)

        self.fc = nn.Linear(self.dim_model, self.dim_output)
        self.fc_ctc = nn.Linear(self.dim_model, self.dim_output)
        self.loss = ce_loss()
        self.ctc = ctc_loss()
        self.weight = config['weight']

    def set_train(self):
        self.prenet.train()
        self.transformer.train()

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
        src = src.cuda()
        with torch.no_grad():
            src_padding_mask = torch.ones(1, src.shape[1], dtype=bool)
            src_padding_mask[:, :src_len]=False
            y = self.prenet(self.enc_pe(src))
            memory = self.transformer.encoder(y.cuda(), src_key_padding_mask=src_padding_mask.cuda())
            ys=torch.ones((1, 1), dtype=torch.int).cuda()
            ys*=2
            memory_mask=None
        for i in range(max_len - 1):
            with torch.no_grad():
                mask=self.generate_square_subsequent_mask(ys.shape[1]).cuda()
                z = self.dec_pe(self.dec_embed(ys))
                z = self.transformer.decoder(z, memory, tgt_mask=mask, memory_mask=memory_mask)
                atts = self.transformer.decoder._get_attention_weights()
                z = F.log_softmax(self.fc(z), dim=-1)
                z = torch.argmax(z[:, -1, :]).reshape(1, 1)

                ys = torch.cat((ys, z), dim=1) #(1, T+1)

                if z == 3:
                    break

            torch.cuda.empty_cache()

        ys = ys.to('cpu').detach().numpy().copy().squeeze()
        return ys.tolist(), atts
