from pyparsing import col
from tqdm import tqdm 

import torch 
from torch.nn.utils.rnn import pad_sequence

from src.dataset.tokenizer import Tokenizer
from src.dataset.base_dataset import Batch

class Inference:
    def __init__(self, config, model, device):
        self.config = config
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        self.tokenizer = Tokenizer(self.config)
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def make_batch(self, input_ids, output_ids):
        src_mask = torch.zeros((input_ids.size(1), input_ids.size(1))).type(torch.bool)
        src_key_padding_mask = input_ids == 0
        tgt_mask = self._generate_square_subsequent_mask(output_ids.size(1))
        tgt_key_padding_mask = output_ids == 0
        memory_mask = torch.zeros((output_ids.size(1), input_ids.size(1))).type(torch.bool) 
        
        return Batch(input_ids, output_ids, src_mask, tgt_mask, memory_mask,
                     src_key_padding_mask, tgt_key_padding_mask, src_key_padding_mask)
        
    def run_infer_on_batch(self, batch, device):
        batch.to_device(device)
        memory = self.model(batch, return_type='encode')  
        
        for _ in range(int(self.config['model']['input_len'])):
            logits = self.model(batch, 'decode', memory)
            next_token_id = torch.argmax(logits, dim=2)[:, -1].unsqueeze(1)
            tgt_ids = torch.cat([batch.tgt_ids, next_token_id], dim=1)
            batch = self.make_batch(batch.input_ids, tgt_ids)
            batch.to_device(device)
            
        generated_seq = batch.tgt_ids.tolist()
        
        return generated_seq

def test_inference_class(config):
    from torch.utils.data import DataLoader
    from src.model.transformer import Seq2SeqModel
    from src.dataset.tokenizer import Tokenizer
    from src.dataset.base_dataset import BaseDataset
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(config)
    dataset = BaseDataset(config, 'data\processed\\test_dataset.txt', tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
    model = Seq2SeqModel(config)
    model.load_state_dict(torch.load(config['data_path']['model_ckpt']))
    inference = Inference(config, model, device)
    sos_id = torch.tensor([[int(tokenizer.vocab['<sos>'])]], device=device)
    
    for i, batch in enumerate(dataloader):
        if i == 5: break
        target = batch.tgt_ids.tolist()
        batch = inference.make_batch(batch.input_ids, sos_id)
        pred = inference.run_infer_on_batch(batch, device)
        
        print(pred)
        print(target)
        pred = [i for i in pred if i != 0]
        if pred == target[0]:
            print('correct')
        print('===============')
        
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
    
    
    
    
        
        
        
        