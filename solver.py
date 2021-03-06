import os,sys
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from generator import SpeechDataset
import metric
import numpy as np
from tqdm import tqdm
import h5py

'''
    train
    繰り返し１回分の学習を行う
'''
def train(model, loader, optimizer, iterm, epoch, writer):
    model.train()

    with tqdm(total=len(loader.dataset), leave=False) as bar:
        for batch_idx, _data in enumerate(loader):
            # バッチ
            inputs, labels, input_lengths, label_lengths, _ = _data
            # listからTensorへの変換
            input_lengths = torch.tensor(input_lengths).to(torch.int32)
            label_lengths = torch.tensor(label_lengths).to(torch.int32)

            optimizer.zero_grad()
            # 音声認識モデルの順伝播を行う
            # 返り値は予測系列と損失
            _, loss = model(inputs.cuda(), labels.cuda(),
                            input_lengths.cuda(), label_lengths.cuda())
            loss.backward()
            optimizer.step()
            iterm.step()

            # tensorboard用に損失を書き出す
            writer.add_scalar('loss', loss.item(), iterm.get())

            # 進捗バーの処理
            bar.set_description("[Epoch %d]" % epoch)
            bar.set_postfix_str(f'{loss:.3f}')
            bar.update(len(inputs))

            del loss

            torch.cuda.empty_cache()

'''
    test
    バリデーションとデコード（greedy_decode）を行う
'''
def test(model, loader, iterm, epoch, writer):
    model.eval()

    loss=[]
    cer=[]
    with tqdm(total=len(loader.dataset), leave=False) as bar:
        with torch.no_grad():
            for i, _data in enumerate(loader):
                inputs, labels, input_lengths, label_lengths, _ = _data
                # listからTensorへの変換
                input_lengths = torch.tensor(input_lengths).to(torch.int32)
                label_lengths = torch.tensor(label_lengths).to(torch.int32)

                # バリデーションは順伝播のみを行い損失を得る．
                _, loss_value = model(inputs.cuda(), labels.cuda(),
                                    input_lengths.cuda(), label_lengths.cuda())
                loss.append(loss_value.item())

                # バッチ内のデータを１つずつデコード
                for j in range(inputs.shape[0]):
                    pred, _ = model.greedy_decode(torch.unsqueeze(inputs[j],0),
                                                 input_lengths[j],
                                                 max_len=label_lengths[j])
                    target = labels[j][:label_lengths[j]].tolist()
                    # PER(音素誤り率を計算)
                    cer.append(metric.cer(target, pred))

                # 進捗バーの処理
                bar.set_description("[Epoch %d]" % epoch)
                bar.set_postfix_str(f'{np.mean(loss):.3f} {np.mean(cer):.3f}')
                bar.update(len(inputs))

                # tensorboard用に損失と誤り率の平均を書き出す
                writer.add_scalar('test_loss', np.mean(loss), iterm.get())
                writer.add_scalar('test_cer', np.mean(cer), iterm.get())
    return np.mean(cer)

'''
    transformerによるデコード
    音響特徴量から系列を推定してファイルに保存する．
    デコード時のattention重みも保存する．
'''
def decode(model, loader, vocab, config):
    model.eval()

    pers=[]
    with open(config['decode'], 'w') as f:
        with h5py.File(config['attention'], 'w') as h5f:
            with torch.no_grad():
                for i, _data in enumerate(loader):
                    inputs, labels, input_lengths, label_lengths, keys = _data
                    input_lengths = torch.tensor(input_lengths).to(torch.int32)
                    label_lengths = torch.tensor(label_lengths).to(torch.int32)

                    for j in range(inputs.shape[0]):
                        pred, atts = model.greedy_decode(torch.unsqueeze(inputs[j],0),
                                                    input_lengths[j],
                                                    max_len=label_lengths[j])
                        pred = vocab.remove_syms(pred)
                        target = vocab.remove_syms(labels[j][:label_lengths[j]].tolist())

                        # 音素誤り率(Phone Error Rate; PER)を計算
                        per=metric.cer(target, pred)
                        pers.append(per)

                        text=" ".join(vocab.to_string(target))
                        f.write(f'{keys[j]} REF: {text}\n')
                        text=" ".join(vocab.to_string(pred))
                        f.write(f'{keys[j]} REF: {text}\n')
                        f.write(f'PER: {per:.3f}\n\n')

                        h5f.create_group(keys[j])
                        for n, att in enumerate(atts):
                            h5f.create_dataset(keys[j]+'/attention_weight_'+str(n+1), data=att, compression='gzip', compression_opts=9)
