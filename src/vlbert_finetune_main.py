import logging
import os
import sys
import time
import torch

import json
import shutil
from category_id_map import *
from third_party.ema import *
from third_party.fgm import *
import numpy as np

from vlbert_finetune_model import MultiModal
from vlbert_finetune_config import parse_args
from finetune_data_helper import create_dataloaders
from util import setup_device, setup_seed, setup_logging, build_optimizer, evaluate
from sklearn.model_selection import StratifiedKFold
from collections import Counter

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
    
    # 1. load data
    if (args.use_val):
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
    model = MultiModal(args)
    
    if (args.pretrain_ckpt_file):
        checkpoint = torch.load(args.pretrain_ckpt_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict = False)
    
    # print(model)
    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    # model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank], output_device=args.local_rank)
    model = torch.nn.DataParallel(model.cuda())
    
    fgm = FGM(model, args.fgm_epsilon)
    ema = EMA(model, args.ema_decay)
    ema.register()

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            # torch.cuda.empty_cache()
            model.train()
            loss, accuracy, _, _ = model(inputs=batch)
            loss = loss.mean()
            accuracy = accuracy.mean()
            loss.backward()
            
            if (args.use_fgm):
                fgm.attack() 
                loss_sum, _, _, _ = model(inputs=batch)
                loss_sum = loss_sum.mean()
                loss_sum.backward()
                fgm.restore()
            
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

            if (step >= args.minimum_val_steps and step % args.val_steps == 0):

                if (args.use_ema):
                    ema.apply_shadow()
                
                if (args.use_val):
                    # 4. validation
                    loss, results = validate(model, val_dataloader)
                    results = {k: round(v, 4) for k, v in results.items()}
                    logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")
                        
                if (step == args.save_steps):
                    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                    torch.save({'epoch': epoch, 'step': step, 'model_state_dict': state_dict}, f'save/{args.model_name}/model_step_{step}.bin')
                    sys.exit()
                    
                if (args.use_ema):
                    ema.restore()

                    
def main():
    args = parse_args()
    
    os.makedirs('records/', exist_ok = True)
    os.makedirs('save/' + args.model_name, exist_ok=True)
    
    setup_logging(args)
    setup_device(args)
    setup_seed(args)
    logging.info("Training/evaluation parameters: %s", args)
    
    train_and_validate(args)


if __name__ == '__main__':
    main()
