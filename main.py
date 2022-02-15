import configparser
import os

def run_train(config):
    from src.dataset.base_dataset import BaseDataset
    from src.dataset.tokenizer import Tokenizer
    from src.model.transformer import Seq2SeqModel
    from src.trainer import Trainer
    tokenizer = Tokenizer(config)
    val_dataset = BaseDataset(config, 'data\processed\\val_dataset.txt', tokenizer)
    train_dataset = BaseDataset(config, 'data\processed\\train_dataset.txt', tokenizer)
    model = Seq2SeqModel(config)
    
    trainer = Trainer(config, model, train_dataset, val_dataset)
    trainer.run_train()

def run_test(config):
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
    dataset = BaseDataset(config, 'data\processed\\test_dataset.txt', tokenizer)
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
    
    # run_train(config)
    run_test(config)
    
    
    
    