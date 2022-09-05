import torch
import tqdm
from torch.utils.data import SequentialSampler, DataLoader, DistributedSampler, Subset, Sampler
from torch.cuda.amp import autocast

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

from vlbert_finetune_config import parse_args
from finetune_data_helper import MultiModalDataset
from category_id_map import lv2id_to_category_id
from vlbert_finetune_model import MultiModal
from util import setup_device, setup_seed

import time
import os
import numpy as np

class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.
    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized, start_idx, end_idx) -> None:
        self.data_source = data_source
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start_idx, self.end_idx))

    def __len__(self) -> int:
        return (self.end_idx - self.start_idx)

def inference():
    
    start = time.time()
    
    args = parse_args()
    setup_device(args)
    setup_seed(args)
    
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    
    args.device = f'cuda:{args.local_rank}'
    
    dataset = MultiModalDataset(args, args.test_annotation, args.test_zip_frames, args.test_zip_feat_path + '_' + str(args.local_rank) + '.zip', test_mode=True)
    
    model = MultiModal(args, test_mode=True)
    
    checkpoint = torch.load(args.ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # 1. load data
    if (args.local_rank == 0):
        # subdataset = Subset(dataset, range(0,len(dataset)//2))
        sampler = SubsetSequentialSampler(dataset, 0, len(dataset)//2)
    else:
        # subdataset = Subset(dataset, range(len(dataset)//2, len(dataset)))
        sampler = SubsetSequentialSampler(dataset, len(dataset)//2, len(dataset))
        
    # sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,
                            batch_size=args.test_batch_size,
                            sampler=sampler,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=args.test_num_workers,
                            prefetch_factor=args.test_prefetch_factor)

    model = torch.nn.parallel.DistributedDataParallel(model.cuda().half())
    # model = torch.nn.parallel.DataParallel(model.cuda())
    model.eval()

    # 3. inference
    predictions = []
    prediction_probs = np.empty((0,200))
    
    with torch.no_grad():
        for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
            # with autocast():
            pred_label_id, pred_prob = model(inputs=batch, inference=True)
            predictions.extend(pred_label_id.cpu().numpy())
            prediction_probs = np.vstack((prediction_probs,pred_prob.cpu().numpy()))
            
    os.makedirs(args.test_output_path, exist_ok=True)
    
    # 4. dump results
    if (args.local_rank == 0):
        with open(f'{args.test_output_path}/result_label.csv', 'a+') as f:
            for pred_label_id, ann in zip(predictions, dataset.anns[:len(dataset)//2]):
                video_id = ann['id']
                category_id = lv2id_to_category_id(pred_label_id)
                f.write(f'{video_id},{category_id}\n')
                
        with open(f'{args.test_output_path}/result_prob_0.npy', 'wb') as f:
            np.save(f, prediction_probs)
        
    else:
        with open(f'{args.test_output_path}/result_label.csv', 'a+') as f:
            for pred_label_id, ann in zip(predictions, dataset.anns[len(dataset)//2:]):
                video_id = ann['id']
                category_id = lv2id_to_category_id(pred_label_id)
                f.write(f'{video_id},{category_id}\n')
                
        with open(f'{args.test_output_path}/result_prob_1.npy', 'wb') as f:
            np.save(f, prediction_probs)
                
    end = time.time()
    
    if (args.local_rank == 0):
        print (end-start)
        # with open('io_results.txt', 'a+') as f:
        #     f.write(f'{args.test_batch_size},{args.test_num_workers},{args.test_prefetch_factor},{end-start}\n')

if __name__ == '__main__':
   
    inference()

