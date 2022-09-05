import logging
import os
import sys
import time
import torch
import shutil
import json
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup,  get_cosine_schedule_with_warmup

from vlbert_pretrain_config import parse_args
from pretrain_data_helper import create_dataloaders
from vlbert_pretrain_model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer
from collections import Counter

def train_and_validate(args):

    train_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    
    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model,num_total_steps)
    model = torch.nn.parallel.DataParallel(model.cuda())
        
    # print(model)
    
    # 3. training
    step = 0
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:
            
            model.train()
            
            if (args.use_mfm):
                mlm_loss, mfm_loss, itm_loss, loss = model(batch)
                mlm_loss = mlm_loss.mean()
                mfm_loss = mfm_loss.mean()
                itm_loss = itm_loss.mean()                
            else:
                mlm_loss, itm_loss, loss = model(batch)
                mlm_loss = mlm_loss.mean()
                itm_loss = itm_loss.mean()                
                
            loss = loss.mean()
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            
            if step % args.print_steps == 0:
                time_per_step = (time.time() - start_time) / max(1, step)
                remaining_time = time_per_step * (num_total_steps - step)
                remaining_time = time.strftime('%d:%H:%M:%S', time.gmtime(remaining_time))
                if (args.use_mfm):
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f} mlm_loss {mlm_loss:.3f} mfm_loss {mfm_loss:.3f} itm_loss {itm_loss:.3f} ")
                else:
                    logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f} mlm_loss {mlm_loss:.3f} itm_loss {itm_loss:.3f} ")                    
            if (step == args.save_steps):
                state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
                torch.save({'epoch': epoch, 'step': step, 'model_state_dict': state_dict}, f'save/{args.model_name}/model_step_{step}.bin')
                sys.exit()

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
