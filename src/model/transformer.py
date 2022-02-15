import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelOutput:
    # Output from model
    loss: torch.Tensor
    logits: torch.Tensor

class Seq2SeqModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqModel, self).__init__()
        self.config = config['model']
        self.word_emb = nn.Embedding(int(self.config['vocab_size']), 
                                     int(self.config['hidden_dim'])
                                     )
        self.pos_emb = nn.Embedding(int(self.config['input_len']),
                                    int(self.config['hidden_dim'])
                                    )
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=int(self.config['hidden_dim']),
                                                        nhead=int(self.config['n_attn_heads']),
                                                        dim_feedforward=int(self.config['hidden_dim']),
                                                        dropout=float(self.config['dropout']),
                                                        batch_first=True,
                                                        norm_first=self.config.getboolean('norm_first')
                                                        )
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer,
                                             num_layers=int(self.config['n_layers']))
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=int(self.config['hidden_dim']),
                                                        nhead=int(self.config['n_attn_heads']),
                                                        dim_feedforward=int(self.config['hidden_dim']),
                                                        dropout=float(self.config['dropout']),
                                                        batch_first=True,
                                                        norm_first=self.config.getboolean('norm_first')
                                                        )
        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer,
                                             num_layers=int(self.config['n_layers']))
        
        self.lm_head = nn.Linear(int(self.config['hidden_dim']),
                                 int(self.config['vocab_size']))
        
    def loss_fn(self, logits, tgt_ids):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt_ids[..., 1:].contiguous()
        
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def embedding(self, id_tensor):
        position_ids = torch.arange(0, id_tensor.size(-1), dtype=torch.long, device=id_tensor.device)
        w_embedding = self.word_emb(id_tensor)
        p_embedding = self.pos_emb(position_ids)
        embedding = w_embedding + p_embedding
        
        return embedding
    
    def forward(self, batch, return_type='loss', memory=None):
        if return_type == 'loss':
            memory = self._encode(batch)
            logits = self._decode(batch, memory)
            
            loss = self.loss_fn(logits, batch.tgt_ids)
            
            return loss
        
        elif return_type == 'encode':
            memory = self._encode(batch)
            
            return memory
        
        elif return_type == 'decode':
            logits = self._decode(batch, memory)
            
            return logits
    
    def _encode(self, batch):
        src_embedding = self.embedding(batch.input_ids)
        memory = self.encoder(src_embedding, batch.src_mask)
        
        return memory
    
    def _decode(self, batch, memory):
        tgt_embedding = self.embedding(batch.tgt_ids)
        hidden = self.decoder(tgt_embedding, memory, batch.tgt_mask, batch.memory_mask)
        
        logits = self.lm_head(hidden)
        
        return logits
    

def model_test(config):
    print('starts model test')
    from torch.utils.data import DataLoader
    from src.dataset.base_dataset import BaseDataset
    from src.dataset.tokenizer import Tokenizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(config)
    dataset = BaseDataset(config, 'data\processed\\val_dataset.txt', tokenizer)
    dataloader = DataLoader(dataset, 
                            batch_size=4,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)
    model = Seq2SeqModel(config)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable params: {total_params}')
    
    # with open('network.txt', 'w') as f:
    #     for name, param in model.named_parameters():
    #         if param.requires_grad:
    #             f.write(name + ' ' + str(param.numel()) + '\n')
    
    for i, batch in enumerate(dataloader):
        if i == 5: break
        batch.to_device(device)
        out = model(batch, return_type='loss')
        print(out)
        print(out.size())
        print('==============================')
    

if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    model_test(config)
    
    