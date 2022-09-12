import logging
import os
import time
import torch
import shutil
import json
from category_id_map import *
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup,  get_cosine_schedule_with_warmup

from albef_pretrain_config import parse_args
from pretrain_data_helper import create_dataloaders
from albef_pretrain_model import ALBEF
from util import setup_device, setup_seed, setup_logging
from collections import Counter

def build_optimizer(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = []

    #Layer decay learning rate

    layers =  list(getattr(model, 'bert').encoder.layer)
    layers.reverse()

    lr = args.bert_learning_rate
    for layer in layers:
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

    optimizer_grouped_parameters += [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert.embeddings' in n], 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert.embeddings' in n], 'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (not 'bert' in n)],
'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and (not 'bert' in n)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio*num_total_steps),
                                                num_training_steps=num_total_steps)
    return optimizer, scheduler

def train_and_validate(args):
    train_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = ALBEF(args)
    num_total_steps = len(train_dataloader) * args.max_epochs
    optimizer, scheduler = build_optimizer(args, model, num_total_steps)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # print(model)

    # 3. training
    step = 0
    start_time = time.time()
    
    for epoch in range(args.max_epochs):
        for batch in train_dataloader:

            model.train()
            mlm_loss, mfm_loss, itm_loss, loss = model(batch)

            if args.use_mlm:
                mlm_loss = mlm_loss.mean()

            if args.use_itm:
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
                logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f} mlm_loss {mlm_loss:.3f} mfm_loss {mfm_loss:.3f} itm_loss {itm_loss:.3f} ")

            if (step == args.save_steps):
                # 5. save checkpoint
                torch.save({'model_state_dict': model.module.state_dict()}, f'data/pretrain_save/{args.savedmodel_path}/model_step_{step}.bin')
                return


def main():
    args = parse_args()
    
    if (not args.use_parallel):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    setup_logging(args)
    setup_device(args)
    setup_seed(args)

    os.makedirs('data/pretrain_save/'+args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)
    train_and_validate(args)


if __name__ == '__main__':
    main()
