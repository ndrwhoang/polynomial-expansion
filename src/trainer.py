import random
import numpy as np
import os
import json
from tqdm import tqdm 
import logging
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# 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
class Trainer:
    def __init__(self, config, model, train_dataset, dev_dataset=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = train_dataset
        if dev_dataset is not None:
            self.dev_dataset = dev_dataset
            
        # Seed 
        self.set_seed()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'---------- device: {self.device}')
        
        self.model.to(self.device)            
        # Create dataloaders
        self.train_loader, self.dev_loader = self._get_dataloader(self.config)
        
        # Create optimizer
        self.optimizer = self._get_optimizer(self.config) 
    
    def set_seed(self):
        self.seed = int(self.config['general']['seed'])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _get_dataloader(self, config):
        train_loader = DataLoader(self.train_dataset,
                                      batch_size=int(config['training']['train_bsz']),
                                      collate_fn=self.train_dataset.collate_fn,
                                      shuffle=True,
                                      drop_last=True
                                      )
        dev_loader = DataLoader(self.dev_dataset,
                                    batch_size=int(config['training']['val_bsz']),
                                    collate_fn=self.dev_dataset.collate_fn,
                                    shuffle=False,
                                    drop_last=False
                                    )
        
        return train_loader, dev_loader
    
    def _get_optimizer(self, config):
        model_params = list(self.model.named_parameters())
        no_decay = ['bias']
        optimized_params = [
            {
                'params':[p for n, p in model_params if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }   
        ]
        optimizer = AdamW(optimized_params, lr=float(config['training']['lr']))
        
        return optimizer
    
    def run_train(self):
        best_val_loss = self.run_validation()
        
        for epoch in range(int(self.config['training']['n_epochs'])):
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            batch_loss = 0
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            for i, batch in pbar:
                batch_loss = self._training_step(batch)
                
                # step
                self.optimizer.step()
                self.model.zero_grad(set_to_none=True)
                
                pbar.set_description(f'(Training) Epoch: {epoch} - Steps: {i}/{len(self.train_loader)} - Loss: {batch_loss}')
            
            val_loss = self.run_validation()
            
            if val_loss < best_val_loss:
                logger.info(f'New best validatoin loss at {val_loss}, saving checkpoint')
                best_val_loss = val_loss
                ckpt_path = os.path.join(self.config['training']['model_dir'], 'seq2seq_256_8_3.pt')
                torch.save(self.model.state_dict(), ckpt_path)
                logger.info(f'New checkpoint saved at {ckpt_path}')
        
    def run_validation(self):
        pbar = tqdm(enumerate(self.dev_loader), total=len(self.dev_loader))
        self.model.eval()
        epoch_loss = 0
        
        for i, batch in pbar:
            loss = self._prediction_step(batch)
            pbar.set_description(f'(Validating) Steps: {i}/{len(self.dev_loader)} - Loss: {loss}')
            epoch_loss += loss
        
        logger.info(f' Validation loss: {epoch_loss}')
        
        return epoch_loss
       
    def _training_step(self, batch):
        batch.to_device(self.device)
        loss = self.model(batch)
        loss.backward()
        
        return loss.detach()
    
    @torch.no_grad()
    def _prediction_step(self, batch):
        batch.to_device(self.device)
        loss = self.model(batch)
        
        return loss.detach()

def trainer_test(config):
    from torch.utils.data import DataLoader
    from src.dataset.base_dataset import BaseDataset
    from src.dataset.tokenizer import Tokenizer
    from src.model.transformer import Seq2SeqModel
    
    tokenizer = Tokenizer(config)
    val_dataset = BaseDataset(config, 'data\processed\\train_dataset.txt', tokenizer)
    train_dataset = BaseDataset(config, 'data\processed\\val_dataset.txt', tokenizer)
    model = Seq2SeqModel(config)
    
    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.run_train()
    
if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    trainer_test(config)