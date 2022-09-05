import os
import io
import json
import torch
import tqdm
import zipfile
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

from torch.utils.data import Dataset, SequentialSampler, DataLoader, DistributedSampler, Subset, Sampler
from torch.cuda.amp import autocast

from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from third_party.swin2 import swin_tiny, swin_base
import torch.nn as nn
import time

from util import setup_device, setup_seed

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        if (args.model_pretrained_path):
            self.visual_backbone = swin_tiny()
        else:
            self.visual_backbone = swin_tiny(args.swin_pretrained_path)

    def forward(self, inputs):
        vision_input_embedding = self.visual_backbone(inputs.cuda())
        return vision_input_embedding
    
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
    
class RawFrameDataset(Dataset):

    def __init__(self,
                 args,
                 ann_path: str,
                 zip_frame_dir: str):
        """ This class is used to load raw video frames.
        Args:
            ann_paths (str): the annotation file path.
            zip_frame_dir (str): the directory that saves zip frames.
            max_video_frames (str): the maximum number of video frames.
        """
        
        self.max_frame = args.max_frames
        
        # load annotations
        with open(ann_path, 'r', encoding='utf8') as f:
            self.anns = json.load(f)
        self.zip_frame_dir = zip_frame_dir
        
        # we follow the common practice as in the ImageNet's preprocessing.
        self.transform = Compose([
                # Resize(256),
                CenterCrop(256),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
            ])

        # self.mean_tensor = torch.HalfTensor([0.485, 0.456, 0.406])
        # self.std_tensor = torch.HalfTensor([0.229, 0.224, 0.225])
        
    def __len__(self) -> dict:
        return len(self.anns)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """ Extract the frame tensor from zipped file.
        The output tensor is in shape of [MAX_FRAMES, 3, 224, 224]
        """
        
        # read data from zipfile
        vid = self.anns[idx]['id']
        zip_path = os.path.join(self.zip_frame_dir, f'{vid[-3:]}/{vid}.zip')
        handler = zipfile.ZipFile(zip_path, 'r')
        namelist = sorted(handler.namelist())

        num_frames = len(namelist)
        
        if num_frames <= self.max_frame:
            # load all frame
            select_inds = list(range(num_frames))
        else:
            # if the number of frames exceeds the limitation, we need to sample
            # the frames.
            # uniformly sample when test mode is True
            step = num_frames // self.max_frame
            select_inds = list(range(0, num_frames, step))
            select_inds = select_inds[:self.max_frame]
        
        frame = torch.zeros((self.max_frame, 3, 256, 256), dtype=torch.float16)
        
        for i, j in enumerate(select_inds):
            
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
            
            frame[i] = img_tensor.half()
            
        return dict(img=frame, num_frames=len(select_inds))
        
def parse_args():
    
    parser = argparse.ArgumentParser("Visual feature extraction")
    
    parser.add_argument('--zip_frame_dir', type=str, default='/opt/ml/input/data/zip_frames/labeled/')
    parser.add_argument('--ann_path', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    
    parser.add_argument('--swin_pretrained_path', type=str, default='/opt/ml/wxcode/opensource_models/swinv2_tiny_patch4_window8_256.pth')
    # parser.add_argument('--model_pretrained_path', type=str, default='save/model.bin')
    parser.add_argument('--model_pretrained_path', type=str, default=None)
    
    parser.add_argument('--output_path', type=str, default='temp')
    parser.add_argument('--max_frames', type=int, default=16)
    
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--prefetch_factor', default=2, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")
    
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--seed", type=int, default=2022, help="random seed.")

    args = parser.parse_args()
    return args

def main():
    
    # torch.multiprocessing.set_start_method('spawn')
    
    start = time.time()
    
    args = parse_args()
    os.makedirs(args.output_path, exist_ok = True)
    
    setup_device(args)
    setup_seed(args)
    
    dataset = RawFrameDataset(args, args.ann_path, args.zip_frame_dir)
    
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    args.device = f'cuda:{args.local_rank}'
    
    model = MultiModal(args)
    
    if (args.model_pretrained_path):
        checkpoint = torch.load(args.model_pretrained_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = torch.nn.parallel.DistributedDataParallel(model.cuda().half())        
    model.eval()

    if (args.local_rank == 0):
        # subdataset = Subset(dataset, range(0,len(dataset)//2))
        sampler = SubsetSequentialSampler(dataset, 0, len(dataset)//2)
    else:
        # subdataset = Subset(dataset, range(len(dataset)//2, len(dataset)))
        sampler = SubsetSequentialSampler(dataset, len(dataset)//2, len(dataset))
        
    # batch-size == 8 is fine for V100 GPU, please consider use smaller batch-size if OOM issue occurs.
    dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        sampler=sampler,
                        drop_last=False,
                        pin_memory=True,
                        num_workers=args.num_workers,
                        prefetch_factor=args.prefetch_factor)
        
    # assert not os.path.isfile(args.output_path), f"{args.output_path} already exists. " \
    #                                               "If you want to override it, please manually delete this file."
    
    if os.path.exists(args.output_path + '_' + str(args.local_rank) + '.zip'):
        os.remove(args.output_path + '_' + str(args.local_rank) + '.zip')
        
    output_handler = zipfile.ZipFile(args.output_path + '_' + str(args.local_rank) + '.zip', 'w', compression=zipfile.ZIP_STORED)

    with torch.no_grad():
        
        if (args.local_rank == 0):
            cur = 0
        else:
            cur = len(dataset)//2
        
        for batch_id, dataitem in tqdm.tqdm(enumerate(dataloader)):
            img, num_frames = dataitem['img'], dataitem['num_frames']
            B, L = img.shape[0:2]
            img = img.view((B * L, ) + img.shape[2:])
            # with autocast():
            feature = model(img)
            feature = feature.view(B, L, -1)
            # feature = feature.cpu().numpy().astype(np.float16)
            feature = feature.cpu().numpy()
            for i in range(B):
                feedid = dataset.anns[cur]['id']
                ioproxy = io.BytesIO()
                np.save(ioproxy, feature[i, :int(num_frames[i])])
                npy_str = ioproxy.getvalue()
                output_handler.writestr(f'{feedid}.npy', npy_str)
                cur += 1
            
    output_handler.close()
    
    end = time.time()
    
    if (args.local_rank == 0):
        print (end-start)

if __name__ == '__main__':
    
    main()

