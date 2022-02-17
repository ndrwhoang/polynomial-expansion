from distutils.command.config import config
import re
import os
import logging
import configparser
from tqdm import tqdm
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn import Transformer

from src.dataset.tokenizer import Tokenizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class Batch:
    input_ids: torch.Tensor
    tgt_ids: torch.Tensor
    src_mask: torch.Tensor
    tgt_mask: torch.Tensor
    memory_mask: torch.Tensor
    src_key_padding_mask: torch.Tensor
    tgt_key_padding_mask: torch.Tensor
    memory_key_padding_mask: torch.Tensor

    def to_device(self, device):
        self.input_ids = self.input_ids.to(device)
        self.tgt_ids = self.tgt_ids.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        self.memory_mask = self.memory_mask.to(device)
        self.src_key_padding_mask = self.src_key_padding_mask.to(device)
        self.tgt_key_padding_mask = self.tgt_key_padding_mask.to(device)
        self.memory_key_padding_mask = self.memory_key_padding_mask.to(device)

class BaseDataset(Dataset):
    def __init__(self, config, data_path, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        data = self.get_data(data_path)
        self.input_ids, self.output_ids = self.convert_samples(data)
        self.n_samples = len(self.input_ids)
        
        logger.info(f'n_samples: {self.n_samples}')
        
    def get_data(self, data_path):
        data_path = os.path.join(*data_path.split('\\'))
        with open(data_path, 'r') as f:
            data = f.read().splitlines()
        
        data = data[:200]
            
        return data
    
    def convert_samples(self, data):
        logger.info('Start processing data to inputs')
        input_ids, output_ids = [], []
        
        for i_sample, sample in enumerate(tqdm(data)):
            factorized, expanded = sample.split('=')
            input_id = self.tokenizer.convert_to_ids(factorized)
            output_id = self.tokenizer.convert_to_ids(expanded)
            
            input_ids.append(torch.tensor(input_id))
            output_ids.append(torch.tensor(output_id))
        
        return input_ids, output_ids
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, item):
        return self.input_ids[item], self.output_ids[item]
    
    def collate_fn(self, batch):
        input_ids, output_ids = zip(*batch)
        
        input_ids = pad_sequence(input_ids, batch_first=True)
        src_mask = torch.zeros((input_ids.size(1), input_ids.size(1))).type(torch.bool)
        src_key_padding_mask = input_ids == 0
        output_ids = pad_sequence(output_ids, batch_first=True)
        tgt_mask = self._generate_square_subsequent_mask(output_ids.size(1))
        tgt_key_padding_mask = output_ids == 0
        memory_mask = torch.zeros((output_ids.size(1), input_ids.size(1))).type(torch.bool)        
        
        return Batch(input_ids, output_ids, src_mask, tgt_mask, memory_mask,
                     src_key_padding_mask, tgt_key_padding_mask, src_key_padding_mask)
        
        
if __name__ == '__main__':
    print('hello world')
    from torch.utils.data import DataLoader
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    tokenizer = Tokenizer(config)
    dataset = BaseDataset(config, 'data\processed\\val_dataset.txt', tokenizer)
    dataloader = DataLoader(dataset,
                            batch_size=2, 
                            shuffle=False, 
                            collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        if i == 5: break
        print(batch.input_ids.size())
        print(batch.tgt_ids.size())
        print(batch.src_mask.size())
        print(batch.tgt_mask.size())
        print(batch.memory_mask.size())
        print(batch.src_key_padding_mask.size())
        print(batch.tgt_key_padding_mask.size())
        print(batch.memory_key_padding_mask.size())    
        print('===========')