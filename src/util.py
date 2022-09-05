import logging
import random
import os

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from category_id_map import lv2id_to_lv1id


def setup_device(args):
    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    
    args.n_gpu = torch.cuda.device_count()
    
    # torch.cuda.set_device(args.local_rank)
    # torch.distributed.init_process_group(backend="nccl")
    
    args.device = 'cuda'
    
    print(args.device)


def setup_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logging(args):
    
    # logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S',
    #                     level=logging.INFO)
    
    logging.basicConfig(filename='records/'+args.model_name+'.log',
                        filemode='w',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    
    logger = logging.getLogger(__name__)

    return logger


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
        
    # for n, p in model.named_parameters():
    #     if 'visual_backbone' in n:
    #         p.requires_grad = False
        
    optimizer_grouped_parameters += [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert.embeddings' in n], 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert.embeddings' in n], 'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'vision_bert_embeddings' in n], 'lr': lr, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'vision_bert_embeddings' in n], 'lr': lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'visual_backbone' in n], 'lr': args.swin_learning_rate, 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'visual_backbone' in n], 'lr': args.swin_learning_rate, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if (not any(nd in n for nd in no_decay)) and (not 'bert' in n) and (not 'visual_backbone' in n)],'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if (any(nd in n for nd in no_decay)) and (not 'bert' in n) and (not 'visual_backbone' in n)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.default_learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=args.max_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_ratio*num_total_steps),
                                                num_training_steps=num_total_steps)
    
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.size())
    
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results
