from accelerate import Accelerator

def _run_test(config, test_path):
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
    print('hello world')
    accelerator = Accelerator()