import random
import numpy as np
import os
import json
from tqdm.auto import tqdm 
import logging
from accelerate import Accelerator

import torch
from torch.utils.data import DataLoader
# from torch.optim import AdamW

from transformers import AdamW, get_linear_schedule_with_warmup

logging.basicConfig()
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, accelerator, model, train_dataset, dev_dataset=None):
        self.config = config
        self.accelerator = accelerator
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = train_dataset
        if dev_dataset is not None:
            self.dev_dataset = dev_dataset
            
        self.setup_training()
            
    def setup_training(self):
        # Seed 
        self.set_seed()
        
        # Device
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = self.accelerator.device
        logger.info(f'---------- device: {self.device}')
        
        # self.model.to(self.device)            
        # Create dataloaders
        self.train_loader, self.dev_loader = self._get_dataloader(self.config)
        
        # Create optimizer
        self.optimizer = self._get_optimizer(self.config) 
        
        # Huggingface's Accelerator
        self.model, self.optimizer, self.train_loader, self.dev_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.dev_loader)

        n_warmup_steps = int(self.config['training']['warmup_steps'])
        self.n_grad_accumulation_steps = int(self.config['training']['n_grad_accumulation_steps'])
        n_training_steps = (len(self.train_loader) * int(self.config['training']['n_epochs'])) // self.n_grad_accumulation_steps
        self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                            n_warmup_steps,
                                                            n_training_steps)
    
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
            pbar = tqdm(self.train_loader, disable = not self.accelerator.is_local_main_process)
            # pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            batch_loss = 0
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            for i, batch in enumerate(pbar):
                batch_loss = self._training_step(batch)
                
                # step
                if i % self.n_grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.model.zero_grad(set_to_none=True)
                
                pbar.set_description(f'(Training) Epoch: {epoch} - Steps: {i}/{len(self.train_loader)} - Loss: {batch_loss:.4f}')
            
            val_loss = self.run_validation()
            
            if val_loss < best_val_loss:
                logger.info(f'New best validatoin loss at {val_loss:.4f}, saving checkpoint')
                best_val_loss = val_loss
                self.accelerator.wait_for_everyone()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                ckpt_path = os.path.join(self.config['training']['model_dir'], 'seq2seq_256_8_3.pt')
                # torch.save(unwrapped_model.state_dict(), ckpt_path)
                logger.info(f'New checkpoint saved at {ckpt_path}')
        
    def run_validation(self):
        pbar = tqdm(enumerate(self.dev_loader), total=len(self.dev_loader))
        self.model.eval()
        epoch_loss = 0
        
        for i, batch in pbar:
            loss = self._prediction_step(batch)
            pbar.set_description(f'(Validating) Steps: {i}/{len(self.dev_loader)} - Loss: {loss:.4f}')
            epoch_loss += loss
        
        logger.info(f' Validation loss: {epoch_loss:.4f}')
        
        return epoch_loss
       
    def _training_step(self, batch):
        # batch.to_device(self.device)
        loss = self.model(batch)
        # loss.backward()
        self.accelerator.backward(loss)
        
        return loss.detach()
    
    @torch.no_grad()
    def _prediction_step(self, batch):
        # batch.to_device(self.device)
        loss = self.model(batch)
        
        return loss.detach()

def trainer_test(config, accelerator):
    from src.dataset.base_dataset import BaseDataset
    from src.dataset.tokenizer import Tokenizer
    from src.model.transformer import Seq2SeqModel
    
    tokenizer = Tokenizer(config)
    val_dataset = BaseDataset(config, 'data\processed\\train_subset.txt', tokenizer)
    train_dataset = BaseDataset(config, 'data\processed\\train_subset.txt', tokenizer)
    model = Seq2SeqModel(config)
    
    trainer = Trainer(config, accelerator, model, train_dataset, val_dataset)
    trainer.run_train()
    
if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))   
    
    accelerator = Accelerator(fp16=config['training'].getboolean('fp16'))
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR) 
    logger.info('Accelerator settings')
    logger.info(accelerator.state)
    
    trainer_test(config, accelerator)