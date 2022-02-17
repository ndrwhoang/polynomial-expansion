from tqdm import tqdm

import torch

from src.dataset.base_dataset import Batch

class PredictionMixin:
    def generate_from_batch(self, input_ids, decode_strat='greedy'):
        sos_id = torch.tensor([2]*input_ids.size(0), device=self.device).unsqueeze(1)
        batch = self._prepare_batch(input_ids, sos_id)
        batch.to_device(self.device)
        
        memory = self(batch, return_type='encode')
        
        if decode_strat == 'greedy':
            generated_seq = self._greedy_decode(batch, memory, 32)
        else:
            generated_seq = None

        return generated_seq
    
    def _greedy_decode(self, batch, memory, seq_len):
        for _ in range(int(seq_len)):
            logits = self(batch, 'decode', memory)
            next_token_id = torch.argmax(logits, dim=2)[:, -1].unsqueeze(-1)
            tgt_ids = torch.cat([batch.tgt_ids, next_token_id], dim=1)
            batch = self._prepare_batch(batch.input_ids, tgt_ids)
        
        return batch.tgt_ids.tolist()
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        return mask
    
    def _prepare_batch(self, input_ids, output_ids):
        src_mask = torch.zeros((input_ids.size(1), input_ids.size(1))).type(torch.bool)
        src_key_padding_mask = input_ids == 0
        tgt_mask = self._generate_square_subsequent_mask(output_ids.size(1))
        tgt_key_padding_mask = output_ids == 0
        memory_mask = torch.zeros((output_ids.size(1), input_ids.size(1))).type(torch.bool) 
        out = Batch(input_ids, output_ids, src_mask, tgt_mask, memory_mask,
                     src_key_padding_mask, tgt_key_padding_mask, src_key_padding_mask)
                
        return out
        