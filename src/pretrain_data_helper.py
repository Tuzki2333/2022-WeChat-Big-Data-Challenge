import json
import random
import zipfile
from io import BytesIO

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, SubsetRandomSampler

import transformers
from transformers import BertTokenizer
transformers.logging.set_verbosity_error()
import scipy


def create_dataloaders(args, train_index = None, val_index = None):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_annotation_unlabeled, args.train_zip_feats, args.train_zip_feats_unlabeled)
    
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  drop_last=True,
                                  pin_memory=True,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)
    return train_dataloader


class MultiModalDataset(Dataset):

    def __init__(self,
                 args,
                 ann_path: str,
                 ann_path_unlabeled: str,
                 zip_feats: str,
                 zip_feats_unlabeled: str,
                 test_mode: bool = False):
        
        self.max_frame = args.max_frames
        
        self.bert_seq_length_concat = args.bert_seq_length_concat
        
        self.test_mode = test_mode
        self.num_workers = args.num_workers

        # lazy initialization for zip_handler to avoid multiprocessing-reading error
        self.zip_feat_path = zip_feats
        self.zip_feat_path_unlabeled = zip_feats_unlabeled
        
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles = [None for _ in range(self.num_workers)]
        else:
            self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
            
        if self.num_workers > 0:
            # lazy initialization for zip_handler to avoid multiprocessing-reading error
            self.handles_unlabeled = [None for _ in range(self.num_workers)]
        else:
            self.handles_unlabeled = zipfile.ZipFile(self.zip_feat_path_unlabeled, 'r')

        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
            
        # load annotations
        with open(ann_path_unlabeled, 'r', encoding='utf8') as f:
            self.anns_unlabeled = json.load(f)

        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

    def __len__(self) -> int:
        return len(self.anns) + len(self.anns_unlabeled)

    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        # read data from zipfile
        if (idx < 100000):
            vid = self.anns[idx]['id']
            if self.handles[worker_id] is None:
                self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
            raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        else:
            vid = self.anns_unlabeled[idx-100000]['id']
            if self.handles_unlabeled[worker_id] is None:
                self.handles_unlabeled[worker_id] = zipfile.ZipFile(self.zip_feat_path_unlabeled, 'r')
            raw_feats = np.load(BytesIO(self.handles_unlabeled[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        
        raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape

        feat = np.zeros((self.max_frame, feat_dim), dtype=np.float32)
        mask = np.ones((self.max_frame,), dtype=np.int32)
        if num_frames <= self.max_frame:
            feat[:num_frames] = raw_feats
            mask[num_frames:] = 0
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            if self.test_mode:
                # uniformly sample when test mode is True
                step = num_frames // self.max_frame
                select_inds = list(range(0, num_frames, step))
                select_inds = select_inds[:self.max_frame]
            else:
                # randomly sample when test mode is False
                select_inds = list(range(num_frames))
                random.shuffle(select_inds)
                select_inds = select_inds[:self.max_frame]
                select_inds = sorted(select_inds)
            for i, j in enumerate(select_inds):
                feat[i] = raw_feats[j]
        feat = torch.FloatTensor(feat)
        mask = torch.LongTensor(mask)
        return feat, mask

    def tokenize_sep_text_by_truncate_last(self, text_1: str, text_2: str, text_3: str, max_length: int) -> tuple:
        
        input_ids_1 = self.tokenizer.encode(text_1)[1:-1]
        input_ids_2 = self.tokenizer.encode(text_2)[1:-1]
        input_ids_3 = self.tokenizer.encode(text_3)[1:-1]
        
        length_1 = len(input_ids_1)
        length_2 = len(input_ids_2)
        length_3 = len(input_ids_3)
        
        if (length_1 + length_2 + length_3 > max_length - 5):
            
            length = max_length - 5
            
            if (length_1 + length_2 <= length):
                length_3 = min(length_3, length - length_1 - length_2)
            elif (length_1 <= length):
                length_3 = 0
                length_2 = min(length_2, length - length_1)
            else:
                length_3 = 0
                length_2 = 0
                length_1 = min(length_1, length)
            
        if length_1 == 0:
            input_ids_1 = [101] + [102]  + [102]
            length_1 += 3
        else:
            input_ids_1 = [101] + [102] + input_ids_1[:length_1]  + [102]
            length_1 += 3
        
        if length_2 == 0:
            input_ids_2 = [102]
            length_2 += 1
        else:
            input_ids_2 = input_ids_2[:length_2] + [102]
            length_2 += 1
            
        if length_3 == 0:
            input_ids_3 = [102]
            length_3 += 1
        else:
            input_ids_3 = input_ids_3[-length_3:] + [102]
            length_3 += 1
                
        position_ids = [0] + list(range(33,max_length+32))
        token_type_ids = [0]*2 + [1]*(max_length-2)
                
        input_ids_remain = [0] * (max_length-length_1-length_2-length_3)
        mask = [1]*(length_1+length_2+length_3) + [0]*(max_length-length_1-length_2-length_3)       
        
        return np.array(input_ids_1 + input_ids_2 + input_ids_3 + input_ids_remain), np.array(position_ids), np.array(token_type_ids), np.array(mask)

    def __getitem__(self, idx: int) -> dict:
        
        # Step 1, load visual features from zipfile.
        worker_info = torch.utils.data.get_worker_info()
        frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        if (idx < 100000):
            temp_anns = self.anns[idx]
        else:
            temp_anns = self.anns_unlabeled[idx-100000]
            
        # Step 2, load tokens
        ocr_text = ""
        for i in range(0,len(temp_anns['ocr'])):
            ocr_text += temp_anns['ocr'][i]['text']
        
        concat_text_input, concat_text_position, concat_text_token_type, concat_text_mask = self.tokenize_sep_text_by_truncate_last(temp_anns['title'], temp_anns['asr'], ocr_text, max_length=self.bert_seq_length_concat)
        
        # print(concat_text_input, concat_text_position, concat_text_token_type, concat_text_mask)
        
        # Step 3, summarize into a dictionary
        data = dict(
            frame_input=frame_input,
            frame_mask=frame_mask,

            concat_text_input=concat_text_input,
            concat_text_mask=concat_text_mask
        )

        return data
