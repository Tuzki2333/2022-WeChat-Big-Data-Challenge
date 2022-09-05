import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers import BertTokenizer
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from third_party.qq_pretrain import *

class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = args.device
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        # print(self.bert_config)
        
        bert_output_size = 768
        self.max_frames = args.max_frames
        
        self.vision_fc = MultiLayerPerceptron(args.frame_embedding_size, bert_output_size, bn = False)
        self.vision_bert_embeddings = BertEmbeddings(self.bert_config)
        
        self.use_mlm = True
        self.use_mfm = args.use_mfm
        self.use_itm = True
        
        if self.use_mlm:
            self.lm = MaskWord(tokenizer_path=args.bert_dir, cache_dir=args.bert_cache)
            self.vocab_size = self.bert_config.vocab_size
            self.mlm_head = BertOnlyMLMHead(self.bert_config) 
        
        if self.use_mfm:
            self.vm = MaskVideo()
            self.mfm_head = VisualOnlyMLMHead(self.bert_config) 
            
        if self.use_itm:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(self.bert_config.hidden_size, 1) 

    def forward(self, inputs, inference=False):
        
        mlm_loss, mfm_loss, itm_loss, loss = 0, 0, 0, 0
        
        text_input_ids = inputs['concat_text_input']
        video_feature = inputs['frame_input']
        
        text_input_mask = inputs['concat_text_mask']
        video_input_mask = inputs['frame_mask']
        
        if self.use_mlm:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)
            
        if self.use_mfm:
            vm_input = video_feature
            input_feature, input_mask, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_input_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_input_mask = input_mask.to(video_feature.device)
            video_label = video_label.to(video_feature.device)
            
        if self.use_itm:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)
                
        cls_input_mask = text_input_mask[:, 0:1]
        text_input_mask = text_input_mask[:, 1:]
        
        text_input_embedding = self.bert.embeddings(input_ids = text_input_ids)
        cls_input_embedding = text_input_embedding[:, 0:1, :]
        text_input_embedding = text_input_embedding[:, 1:, :]
            
        vision_input_embedding = self.vision_fc(video_feature)
        vision_input_embedding = self.vision_bert_embeddings(inputs_embeds = vision_input_embedding)

        bert_input_embedding = torch.cat((cls_input_embedding, vision_input_embedding, text_input_embedding), dim = 1)
        bert_mask  = torch.cat((cls_input_mask, video_input_mask, text_input_mask), dim = 1)
        
        bert_output = self.bert.encoder(bert_input_embedding, attention_mask = (1.0 - bert_mask[:,None,None,:]) * -10000.0, output_hidden_states=True)
        bert_output_last_hidden_state = bert_output['last_hidden_state']
        
        # compute pretrain task loss
        if self.use_mlm:
            lm_output = self.mlm_head(bert_output_last_hidden_state)[:,(1+self.max_frames):,:].contiguous().view(-1, self.vocab_size)
            mlm_loss = nn.CrossEntropyLoss()(lm_output, lm_label.contiguous().view(-1))
            loss += mlm_loss /1.25/ (self.use_mlm+self.use_mfm+self.use_itm)
            
        if self.use_mfm:
            vm_output = self.mfm_head(bert_output_last_hidden_state[:,1:(1+self.max_frames),:])
            mfm_loss = calculate_mfm_loss(vm_output, vm_input, video_input_mask, video_label)
            loss += mfm_loss /3/ (self.use_mlm+self.use_mfm+self.use_itm)
            
        if self.use_itm:
            pred = self.newfc_itm(bert_output_last_hidden_state[:,0,:])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss /100/ (self.use_mlm+self.use_mfm+self.use_itm)
        
        if self.use_mfm:
            return mlm_loss, mfm_loss, itm_loss, loss
        else:
            return mlm_loss, itm_loss, loss

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size, embed_sizes = [], last_layer_activation = True, dropout = 0, last_layer_dropout = False, bn = True):
        super().__init__()
        layers = list()
        for embed_size in embed_sizes:
            layers.append(torch.nn.Linear(input_size, embed_size))
            if (bn):
                layers.append(torch.nn.BatchNorm1d(embed_size))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_size = embed_size
            
        layers.append(torch.nn.Linear(input_size, output_size))
        if (last_layer_activation):
            if (bn):
                layers.append(torch.nn.BatchNorm1d(output_size))
            layers.append(torch.nn.ReLU())
            
        if (last_layer_dropout):
            layers.append(torch.nn.Dropout(p=dropout))
        
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
    
