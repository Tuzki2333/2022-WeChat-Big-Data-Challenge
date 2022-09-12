import logging
import os
import time
import torch
import shutil
import json
from category_id_map import *
from third_party.ema import *
from third_party.fgm import *
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup,  get_cosine_schedule_with_warmup

from albef_finetune_config import parse_args
from finetune_data_helper import create_dataloaders
from albef_finetune_model import ALBEF
from util import setup_device, setup_seed, setup_logging, evaluate
from sklearn.model_selection import KFold, StratifiedKFold
from collections import Counter

def build_optimizer(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = []
    
    #Layer decay learning rate
    layers =  list(getattr(model, 'bert').encoder.layer)
    layers.reverse()
    layer_id = len(layers) - 1
    lr = args.bert_learning_rate
    
    for layer in layers:
        if (layer_id >= args.bert_freezing_layers):
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": args.weight_decay,
                    "lr": lr,
                },
                {
                    "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
            lr *= args.bert_layerwise_learning_rate_decay
        else:
            for n, p in layer.named_parameters():
                p.requires_grad = False  
        layer_id -= 1
    optimizer_grouped_parameters += [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert.embeddings' in n], 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert.embeddings' in n], 'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_encoder.embeddings' in n], 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_encoder.embeddings' in n], 'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'vision_fc' in n], 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'vision_fc' in n], 'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (not 'bert' in n) and (not 'vision_fc' in n) and (not 'visual_encoder' in n)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and (not 'bert' in n) and (not 'vision_fc' in n) and (not 'visual_encoder' in n)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio*num_total_steps),
                                                num_training_steps=num_total_steps)
    # for n, p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n, p.size())
    return optimizer, scheduler                
                
def validate(model, val_dataloader):
    model.eval()
    predictions = []
    labels = []
    losses = []
    with torch.no_grad():
        for batch in val_dataloader:
            loss, _, pred_label_id, label = model(batch)
            loss = loss.mean()
            predictions.extend(pred_label_id.cpu().numpy())
            labels.extend(label.cpu().numpy())
            losses.append(loss.cpu().numpy())
    loss = sum(losses) / len(losses)
    results = evaluate(predictions, labels)

    model.train()
    return loss, results


def train_and_validate(args):
    if (args.use_val):
        # 1. load data
        with open(args.train_annotation, 'r', encoding='utf8') as f:
            anns = json.load(f)
            
        x = []
        y = []
        for idx in range(0,len(anns)):
            x.append(idx)
            y.append(category_id_to_lv2id(anns[idx]['category_id']))
        
        cv = StratifiedKFold(n_splits = args.n_splits, shuffle = True, random_state = args.seed)
        
        # print(Counter(np.array(y)))
        
        for fold, (train_index, val_index) in enumerate(cv.split(x, y)):
            
            if (fold == args.fold):
                # print(Counter(np.array(y)[val_index]))
                train_dataloader, val_dataloader = create_dataloaders(args, train_index, val_index)
                break
                
    else:
        train_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    # model = MultiModal(args)
    model = ALBEF(args)
    if (args.pretrain_ckpt_file != ''):
        checkpoint = torch.load(args.pretrain_ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))
        
    # print(model)
    
    fgm = FGM(model)
    ema = EMA(model, 0.999)
    ema.register()
    K = 3
    
    # 3. training
    step = 0
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            model.train()
            loss, accuracy, _, _ = model(batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            
            if (args.use_fgm):
                fgm.attack(0.5, emb_name='bert.embeddings') 
                loss_sum, _, _, _ = model(batch)
                loss_sum.backward() 
                fgm.restore(emb_name='bert.embeddings') 
                
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            if (args.use_ema):
                ema.update()

            step += 1
            
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")

            if (step % args.val_steps == 0 or step == args.save_steps):
                if (args.use_ema):
                    ema.apply_shadow()

                if (args.use_val):
                    # 4. validation
                    loss, results = validate(model, val_dataloader)
                    results = {k: round(v, 4) for k, v in results.items()}
                    logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                
                if (step == args.save_steps):
                    # 5. save checkpoint
                    torch.save({'model_state_dict': model.module.state_dict()}, f'data/finetune_save/{args.savedmodel_path}/model_step_{step}.bin')
                    return
                    
                if (args.use_ema):
                    ema.restore()

def main():
    args = parse_args()
    
    if (not args.use_parallel):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        
    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    os.makedirs('data/finetune_save/'+args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)


if __name__ == '__main__':
    main()
