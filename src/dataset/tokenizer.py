import re
import os
import json
from tqdm import tqdm
from typing import List, Dict


class Tokenizer:
    def __init__(self, config: Dict[str, Dict[str, str]]):
        self.token_pattern = 'sin|cos|tan|\w|\d|\(|\)|\+|-|\*'
        self.vocab = self._load_vocab(config['data_path']['vocab'])

    def _load_vocab(self, path: str) -> Dict[str, str]:
        path = os.path.join(*path.split('\\'))
        # path = 'data\\resources\\vocab.json'
        with open(path, 'r') as f:
            vocab = json.load(f)
        
        return vocab
    
    def tokenize(self, string_in: str) -> List[str]:
        out = re.findall(self.token_pattern, string_in)
        return out
    
    def convert_to_ids(self, string_in: str) -> List[int]:
        tokens = self.tokenize(string_in)
        tokens = ['<sos>'] + tokens + ['<eos>']
        ids_out = [self.vocab.get(token, 1) for token in tokens]
        
        return ids_out

def make_vocab():
    with open('data\processed\\train_dataset.txt', 'r') as f:
        data = f.read().splitlines()
        
    trigs = 'sin|cos|tan'
    variables = '\w'
    coefs = '\d'
    symbols = '\(|\)|\+|-|\*+'
    
    token_pattern = '|'.join([trigs, variables, coefs, symbols])
    
    vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3}
    count = len(vocab)
    
    for i, sample in enumerate(tqdm(data)):
        factorized, expanded = sample.split('=')
        f_tokens = re.findall(token_pattern, factorized)
        e_words = re.findall(token_pattern, expanded)
        
        new_vocab = set(f_tokens + e_words)
        for token in new_vocab:
            if token not in vocab:
                vocab[token] = count
                count += 1
                
    with open('data\\resources\\vocab.json', 'w') as f:
        json.dump(vocab, f)

if __name__ == '__main__':
    import configparser
    
    config = configparser.ConfigParser()
    config.read('configs\config.cfg')
    print('hello world')
    dummy_input = '(-3*s-13)*(2*s+8)=-6*s**2-50*s-104'
    tokenizer = Tokenizer(config)
    print(tokenizer.tokenize)
    
    
    