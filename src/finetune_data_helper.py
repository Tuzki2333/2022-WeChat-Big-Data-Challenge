import os
import json
import zipfile
import random
import torch

from PIL import Image
# import jpeg4py as jpeg
import cv2

from io import BytesIO
from functools import partial
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler, SubsetRandomSampler

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from category_id_map import category_id_to_lv2id

import numpy as np


def create_dataloaders(args, train_index = None, val_index = None):
    dataset = MultiModalDataset(args, args.train_annotation, args.train_zip_frames, args.train_zip_feat_path)

    if args.train_num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.train_num_workers, prefetch_factor=args.train_prefetch_factor)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    if (args.use_val):
        train_sampler = SubsetRandomSampler(train_index)
        val_sampler = SubsetRandomSampler(val_index)
        train_dataloader = dataloader_class(dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_sampler,
                                      drop_last=True)
        val_dataloader = dataloader_class(dataset,
                                    batch_size=args.val_batch_size,
                                    sampler=val_sampler,
                                    drop_last=False)
        return train_dataloader, val_dataloader
    
    else:
        train_sampler = RandomSampler(dataset)
        train_dataloader = dataloader_class(dataset,
                                      batch_size=args.batch_size,
                                      sampler=train_sampler,
                                      drop_last=False)
        return train_dataloader

    
class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.

    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_frame_dir (str): visual frame zip file path.
        test_mode (bool): if it's for testing.

    """

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_frame_dir: str,
                 zip_feat_path: str = None,
                 test_mode: bool = False):
        
        self.max_frame = args.max_frames
        self.bert_seq_length = args.bert_seq_length
        self.end_to_end_mode = args.end_to_end_mode
        self.test_mode = test_mode

        self.zip_frame_dir = zip_frame_dir
        self.zip_feat_path = zip_feat_path
        
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)

        # we use the standard image transform as in the offifical Swin-Transformer.
        self.transform = Compose([
            # Resize(256),
            CenterCrop(256),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        if (not self.end_to_end_mode):
            
            if (self.test_mode):
                if args.test_num_workers > 0:
                    # lazy initialization for zip_handler to avoid multiprocessing-reading error
                    self.handles = [None for _ in range(args.test_num_workers)]
                else:
                    self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')
                    
            else:
                if args.train_num_workers > 0:
                    # lazy initialization for zip_handler to avoid multiprocessing-reading error
                    self.handles = [None for _ in range(args.train_num_workers)]
                else:
                    self.handles = zipfile.ZipFile(self.zip_feat_path, 'r')                

    def __len__(self) -> int:
        return len(self.anns)

    def get_visual_frames(self, idx: int) -> tuple:
        
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        frame = torch.zeros((self.max_frame, 3, 256, 256), dtype=torch.float32)
        mask = torch.zeros((self.max_frame, ), dtype=torch.long)
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
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
            
            mask[i] = 1
            img_content = handler.read(namelist[j])

            img_decode = BytesIO(img_content)
            img_decode = Image.open(img_decode)
            img_tensor = self.transform(img_decode)
            
            # img_decode = jpeg.JPEG(np.frombuffer(img_content, np.uint8)).decode()
            # img_decode = Image.fromarray(img_decode)
            
            # img_decode = cv2.imdecode(np.frombuffer(img_content, np.uint8), 1)
            # # img = cv2.resize(img_decode, (256, 256), cv2.INTER_LINEAR)
            # img = np.array(img_decode[(img_decode.shape[0]//2-128):(img_decode.shape[0]//2+128), (img_decode.shape[1]//2-128):(img_decode.shape[1]//2+128), ::-1], dtype=np.float32)
            # img = img / 255
            # img = img - np.array([0.485, 0.456, 0.406], dtype=np.float32)
            # img = img / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            # img = np.transpose(img, (2, 0, 1))
            # img_tensor = torch.from_numpy(img)
            
            if (self.test_mode):
                frame[i] = img_tensor.half()
            else:
                frame[i] = img_tensor
                
        return frame, mask
    
    def get_visual_feats(self, worker_id, idx: int) -> tuple:
        
        # read data from zipfile
        vid = self.anns[idx]['id']
        if self.handles[worker_id] is None:
            self.handles[worker_id] = zipfile.ZipFile(self.zip_feat_path, 'r')
        raw_feats = np.load(BytesIO(self.handles[worker_id].read(name=f'{vid}.npy')), allow_pickle=True)
        
        if (not self.test_mode):
            raw_feats = raw_feats.astype(np.float32)  # float16 to float32
        num_frames, feat_dim = raw_feats.shape
        
        if (not self.test_mode):
            feat = torch.zeros((self.max_frame, feat_dim), dtype=torch.float32)
        else:
            feat = torch.zeros((self.max_frame, feat_dim), dtype=torch.float16)
            
        mask = torch.ones((self.max_frame, ), dtype=torch.long)
        
        if num_frames <= self.max_frame:
            feat[:num_frames] = torch.from_numpy(raw_feats)
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
                feat[i] = torch.from_numpy(raw_feats[j])
        return feat, mask
    
    # def tokenize_text(self, text: str) -> tuple:
    #     encoded_inputs = self.tokenizer(text, max_length=self.bert_seq_length, padding='max_length', truncation=True)
    #     input_ids = torch.LongTensor(encoded_inputs['input_ids'])
    #     mask = torch.LongTensor(encoded_inputs['attention_mask'])
    #     return input_ids, mask
    
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
        
        if (self.end_to_end_mode):
            frame_input, frame_mask = self.get_visual_frames(idx)
        else:
            worker_info = torch.utils.data.get_worker_info()
            frame_input, frame_mask = self.get_visual_feats(worker_info.id, idx)

        # Step 2, load tokens
        ocr_text = ""
        for i in range(0,len(self.anns[idx]['ocr'])):
            ocr_text += self.anns[idx]['ocr'][i]['text']
            
        concat_text_input, concat_text_position, concat_text_token_type, concat_text_mask = self.tokenize_sep_text_by_truncate_last(self.anns[idx]['title'], self.anns[idx]['asr'], ocr_text, max_length=self.bert_seq_length)
        
        # concat_text_input, concat_text_mask = self.tokenize_text(self.anns[idx]['title'])

        # Step 3, summarize into a dictionary
        # Step 4, load label if not test mode
        
        if not self.test_mode:
            data = dict(
                frame_input=frame_input,
                frame_mask=frame_mask,
                concat_text_input=concat_text_input,
                concat_text_mask=concat_text_mask
            )
            label = category_id_to_lv2id(self.anns[idx]['category_id'])
            data['label'] = torch.LongTensor([label])
            
        else:
            data = dict(
                frame_input=frame_input.half(),
                frame_mask=frame_mask,
                concat_text_input=concat_text_input,
                concat_text_mask=concat_text_mask
            )

        return data
