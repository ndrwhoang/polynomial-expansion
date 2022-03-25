import configparser
import os
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run_train(config, train_path, val_path):
    from src.dataset.base_dataset import BaseDataset
    from src.dataset.tokenizer import Tokenizer
    from src.model.transformer import Seq2SeqModel
    from src.trainer import Trainer
    
    from accelerate import Accelerator
    
    # Init Huggingface training helper
    accelerator = Accelerator(fp16=config['training'].getboolean('fp16'))
    
    # Only log on one process
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR) 
    logger.info('Accelerator settings')
    logger.info(accelerator.state)
    
    # Dataset, Model, Trainer init
    tokenizer = Tokenizer(config)
    val_dataset = BaseDataset(config, val_path, tokenizer)
    train_dataset = BaseDataset(config, train_path, tokenizer)
    model = Seq2SeqModel(config)
    
    trainer = Trainer(config, accelerator, model, train_dataset, val_dataset)
    trainer.run_train()
    
def run_test(config, test_path, model_path):
    from src.model.transformer import Seq2SeqModel
    from src.dataset.tokenizer import Tokenizer
    from src.dataset.base_dataset import BaseDataset
    from src.inference import Inference
    
    from accelerate import Accelerator
    
    # Init Huggingface training helper
    accelerator = Accelerator(fp16=config['training'].getboolean('fp16'))
    
    # Only log on one process
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR) 
    logger.info('Accelerator settings')
    logger.info(accelerator.state)
    
    # Dataset, Model, Inference init
    tokenizer = Tokenizer(config)
    dataset = BaseDataset(config, test_path, tokenizer)
    model = Seq2SeqModel(config)
    
    inference = Inference(config, accelerator, model, dataset, model_path)
    inference.run_test()
    

if __name__ == '__main__':    
    # TODO: make argparse to override config
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    train_path = config['data_path']['train_dataset']
    val_path = config['data_path']['val_dataset']
    test_path = config['data_path']['test_dataset']
    subset_path = config['data_path']['train_subset']    
    model_path = config['data_path']['model_ckpt']
    
    run_train(config, subset_path, subset_path)
    run_test(config, test_path, model_path)
    
    
    
    