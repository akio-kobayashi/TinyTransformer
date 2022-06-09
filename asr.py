import os
import random
import numpy as np
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from generator import SpeechDataset
import generator
from model import ASRModel
from vocab import Vocab
import metric
import solver

def main():

    config = json.load(open('./config.json', 'r'))

    writer = SummaryWriter(log_dir='./data/transformer/log/')
    use_cuda = torch.cuda.is_available()
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    os.environ["PYTHONHASHSEED"] = str(config['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': os.cpu_count(), 'pin_memory': True} if use_cuda else {}

    vocab=Vocab(config)
    dataset=SpeechDataset(config)
    loader =data.DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            shuffle=False,
                            collate_fn=lambda x: generator.data_processing(x),
                            **kwargs)

    model=ASRModel(config)
    print('Number of Parameters: ', sum([param.nelement() for param in model.parameters()]))
    model.load_state_dict(torch.load(config['model'], map_location=torch.device('cpu')))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), config['learning_rate'])
    iterm = metric.IterMeter()

    for epoch in range(1, config['epochs']+1):
        solver.train(model, loader, optimizer, iterm, epoch, writer)
        per = solver.test(model, loader, iterm, epoch, writer)
        print('Epoch %d: PER %.3f' % (epoch, per))
        torch.save(model.to('cpu').state_dict(), config['output'])
        model.to(device)

    solver.decode(model, loader, vocab, config)

if __name__ == "__main__":
    main()
