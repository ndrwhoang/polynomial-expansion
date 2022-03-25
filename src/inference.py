from tqdm.auto import tqdm 
import os
import random 
import numpy as np
import logging

import torch
from torch.utils.data import DataLoader

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 

class Inference:
    def __init__(self, config, accelerator, model, dataset, model_path=None):
        #TODO: make static method to predict 1 sample
        self.config = config
        self.model = model
        self.accelerator = accelerator
        self.tokenizer = dataset.tokenizer
        
        # Setup
        self._set_seed()
        self._init_model(model_path)
        self._init_dataloader(dataset)
    
    def _set_seed(self):
        self.seed = int(self.config['general']['seed'])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def _init_model(self, model_path):
        if model_path:
            path = os.path.join(*model_path.split('\\'))
            self.model.load_state_dict(torch.load(path))
        
        self.model.eval()
        self.model = self.accelerator.prepare(self.model)
        
    def _init_dataloader(self, dataset):
        loader = DataLoader(dataset,
                            batch_size=int(self.config['training']['test_bsz']),
                            collate_fn=dataset.collate_fn,
                            shuffle=False,
                            drop_last=False
                            )
        self.loader = self.accelerator.prepare(loader)
        
    def run_test(self):
        pbar = tqdm(enumerate(self.loader), total=len(self.loader))
        n_right, n_sample = 0, 0
        bsz = int(self.config['training']['test_bsz'])
        out = {'correct': [], 'incorrect': []}
        
        for i, batch in pbar:
            targets = batch.tgt_ids.tolist()
            preds = self.model.generate_from_batch(batch.input_ids)
            n_sample += bsz
            
            
            for target, pred in zip(targets, preds):
                pred = [i for i in pred if i != 0]
                target = [i for i in target if i != 0]
                
                if pred == target:
                    n_right += 1
                    
        logger.info(f' Test set accuracy {n_right/n_sample}')
            
        
def test_inference_mixin(config):
    from torch.utils.data import DataLoader
    from src.model.transformer import Seq2SeqModel
    from src.dataset.tokenizer import Tokenizer
    from src.dataset.base_dataset import BaseDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(config)
    dataset = BaseDataset(config, 'data\processed\\test_dataset.txt', tokenizer)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=dataset.collate_fn)
    model = Seq2SeqModel(config)
    model.load_state_dict(torch.load(config['data_path']['model_ckpt']))
    model.eval()
    
    for i, batch in enumerate(dataloader):
        if i == 5: break
        target = batch.tgt_ids.tolist()
        pred = model.generate_from_batch(batch.input_ids)
        
        for tgt, pred in zip(target, pred):
            pred = [i for i in pred if i != 0]
            tgt = [i for i in tgt if i != 0]
            
            print(pred)
            print(tgt)
            print('============')
    
        
if __name__ == '__main__':
    import configparser
    from torch.utils.data import DataLoader
    from src.model.transformer import Seq2SeqModel
    from src.dataset.tokenizer import Tokenizer
    from src.dataset.base_dataset import BaseDataset
    
    config = configparser.ConfigParser()
    config.read('configs\config.cfg')
    
    test_inference_mixin(config)
    
    
    
    
        
        
        
        