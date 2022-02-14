import configparser
import os


if __name__ == '__main__':
    from src.dataset.base_dataset import BaseDataset
    from src.dataset.tokenizer import Tokenizer
    from src.model.transformer import Seq2SeqModel
    from src.trainer import Trainer
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    tokenizer = Tokenizer(config)
    val_dataset = BaseDataset(config, 'data\processed\\train_dataset.txt', tokenizer)
    train_dataset = BaseDataset(config, 'data\processed\\val_dataset.txt', tokenizer)
    model = Seq2SeqModel(config)
    
    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.run_train()
    