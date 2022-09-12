import math
import torch
from torch import nn
import torch.nn.functional as F
from third_party.qq_pretrain import *
from third_party.xbert import BertConfig, BertModel, BertEmbeddings, BertOnlyMLMHead

import transformers
from transformers import BertTokenizer

from category_id_map import CATEGORY_ID_LIST, LV1ID, lv2id_to_lv1id, lv1id_to_nest
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.bert_config = BertConfig.from_json_file("third_party/bert.json")
        self.vision_config = transformers.BertConfig(num_hidden_layers=args.vision_layer_num, vocab_size=21128)
        self.visual_encoder = transformers.BertModel(self.vision_config)
        self.visual_encoder = randomize_model(self.visual_encoder)
        self.bert = BertModel.from_pretrained(args.bert_dir, config=self.bert_config, add_pooling_layer=False)

        self.device = args.device
        # self.vision_fc = MultiLayerPerceptron(args.frame_embedding_size, args.bert_output_size, bn = False)

        self.use_mlm = args.use_mlm
        self.use_itm = args.use_itm

        if self.use_mlm:
            self.lm = MaskWord(tokenizer_path=args.bert_dir, config=self.bert_config, cache_dir=args.bert_cache)
            self.vocab_size = self.bert_config.vocab_size
            self.mlm_head = BertOnlyMLMHead(self.bert_config)

        if self.use_itm:
            self.sv = ShuffleVideo()
            self.newfc_itm = torch.nn.Linear(self.bert_config.hidden_size, 1)


    def forward(self, inputs, inference=False):

        mlm_loss, mfm_loss, itm_loss, loss = 0, 0, 0, 0
        
        text_input_mask = inputs['concat_text_mask']
        text_input_ids = inputs['concat_text_input']
        video_feature = inputs['frame_input']
        video_input_mask = inputs['frame_mask']

        if self.use_mlm:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)

        if self.use_itm:
            input_feature, video_text_match_label = self.sv.torch_shuf_video(video_feature.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_text_match_label = video_text_match_label.to(video_feature.device)

        # vision_input = self.vision_fc(video_feature)
        vision_input = video_feature
        image_embeds = self.visual_encoder.encoder(vision_input, attention_mask = (1.0 - video_input_mask[:,None,None,:]) * -10000.0)['last_hidden_state']
        output = self.bert(text_input_ids,
                                   attention_mask = text_input_mask,
                                   encoder_hidden_states = image_embeds,
                                   encoder_attention_mask = video_input_mask,
                                   return_dict=True
                                   )
        if self.use_mlm:
            lm_output = self.mlm_head(output.last_hidden_state[:,1:,:]).contiguous().view(-1, self.vocab_size)
            mlm_loss = nn.CrossEntropyLoss()(lm_output, lm_label.contiguous().view(-1))
            loss += mlm_loss / (self.use_mlm+self.use_itm)

        if self.use_itm:
            pred = self.newfc_itm(output.last_hidden_state[:,0,:])
            itm_loss = nn.BCEWithLogitsLoss()(pred.view(-1), video_text_match_label.view(-1))
            loss += itm_loss / (self.use_mlm+self.use_itm)

        return mlm_loss, mfm_loss, itm_loss, loss

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size, embed_sizes=[], last_layer_activation=True, dropout=0,
                 last_layer_dropout=False, bn=True):
        super().__init__()
        layers = list()
        for embed_size in embed_sizes:
            layers.append(torch.nn.Linear(input_size, embed_size))
            if (bn):
                layers.append(torch.nn.BatchNorm1d(embed_size))
            layers.append(torch.nn.ReLU())
            # layers.append(gelu())
            layers.append(torch.nn.Dropout(p=dropout))
            input_size = embed_size

        layers.append(torch.nn.Linear(input_size, output_size))
        if (last_layer_activation):
            if (bn):
                layers.append(torch.nn.BatchNorm1d(output_size))
            layers.append(torch.nn.ReLU())
            # layers.append(gelu())

        if (last_layer_dropout):
            layers.append(torch.nn.Dropout(p=dropout))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

def randomize_model(model):
    for module_ in model.named_modules(): 
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model