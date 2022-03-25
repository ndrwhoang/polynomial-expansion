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

def run_test(config, test_path):
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from src.model.transformer import Seq2SeqModel
    from src.dataset.tokenizer import Tokenizer
    from src.dataset.base_dataset import BaseDataset
    from src.inference import Inference
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsz = int(config['training']['test_bsz'])
    tokenizer = Tokenizer(config)
    dataset = BaseDataset(config, test_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False, drop_last=True, collate_fn=dataset.collate_fn)
    model = Seq2SeqModel(config)
    model.load_state_dict(torch.load(config['data_path']['model_ckpt']))
    inference = Inference(config, model, device)
    batch_sos = [int(tokenizer.vocab['<sos>'])]*bsz
    sos_id = torch.tensor(batch_sos, device=device).unsqueeze(1)

    n_right = 0
    n_sample = 0
    for i, batch in enumerate(tqdm(dataloader)):
        target = batch.tgt_ids.tolist()
        batch = inference.make_batch(batch.input_ids, sos_id)
        preds = inference.run_infer_on_batch(batch, device)
        n_sample += bsz
        
        for tgt, pred in zip(target, preds):
            pred = [i for i in pred if i != 0]
            tgt = [i for i in tgt if i != 0]
            
            if pred == tgt:
                n_right += 1
    
    print(f'Test set accuracy {n_right/n_sample}')
    # 0.849
    

if __name__ == '__main__':    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    train_path = config['data_path']['train_dataset']
    val_path = config['data_path']['val_dataset']
    test_path = config['data_path']['test_dataset']
    subset_path = config['data_path']['train_subset']    
    
    run_train(config, subset_path, subset_path)
    # run_test(config, test_path)
    
    
    
    