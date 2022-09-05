import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings

from third_party.swin2 import swin_tiny, swin_base
from category_id_map import CATEGORY_ID_LIST


class MultiModal(nn.Module):
    def __init__(self, args, test_mode=False):
        
        super().__init__()
        
        self.device = args.device
        self.end_to_end_mode = args.end_to_end_mode
        self.test_mode = test_mode
        
        bert_output_size = 768
        self.max_frames = args.max_frames
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir=args.bert_cache)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        
        if (self.end_to_end_mode):
            self.visual_backbone = swin_tiny(args.swin_pretrained_path)
            
        self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size, output_size=args.vlad_hidden_size, dropout=args.dropout)
        self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
        
        self.vision_bert_embeddings = BertEmbeddings(self.bert_config)
        self.vision_fc = MultiLayerPerceptron(args.frame_embedding_size, bert_output_size, bn = False)
        
        self.classifier = MultiLayerPerceptron(args.vlad_hidden_size + 7*bert_output_size, len(CATEGORY_ID_LIST), embed_sizes = [args.fc_size], last_layer_activation = False, dropout = args.dropout)
        
        # self.classifier = MultiLayerPerceptron(7*bert_output_size, len(CATEGORY_ID_LIST), embed_sizes = [args.fc_size], last_layer_activation = False, dropout = args.dropout)

    def forward(self, inputs, inference=False):
        
        text_input_embedding = self.bert.embeddings(input_ids = inputs['concat_text_input'].cuda())
        cls_input_embedding = text_input_embedding[:, 0:1, :]
        text_input_embedding = text_input_embedding[:, 1:, :]
        
        text_input_mask = inputs['concat_text_mask'].cuda()
        cls_input_mask = text_input_mask[:, 0:1]
        text_input_mask = text_input_mask[:, 1:]
        
        if (self.end_to_end_mode):
            vision_input_embedding = self.visual_backbone(inputs['frame_input'].cuda())
        else:
            vision_input_embedding = inputs['frame_input'].cuda()
            
        vision_input_mask = inputs['frame_mask'].cuda()
        
        vision_bert_input_embedding = self.vision_fc(vision_input_embedding)
        vision_bert_input_embedding = self.vision_bert_embeddings(inputs_embeds = vision_bert_input_embedding)
        
        bert_input_embedding = torch.cat((cls_input_embedding, vision_bert_input_embedding, text_input_embedding), dim = 1)
        bert_mask  = torch.cat((cls_input_mask, vision_input_mask, text_input_mask), dim = 1)
        bert_extended_mask = (1.0 - bert_mask[:,None,None,:]) * -10000.0
        if (self.test_mode):
            bert_extended_mask = bert_extended_mask.half()
        
        bert_output = self.bert.encoder(bert_input_embedding, attention_mask = bert_extended_mask, output_hidden_states=True)
        bert_output_last_hidden_state = bert_output['last_hidden_state']
        
        bert_output_cls = self.bert.pooler(bert_output_last_hidden_state)
        
        bert_output_mean_pooling = (bert_output_last_hidden_state[:,1:,:]*bert_mask[:,1:].unsqueeze(-1)).sum(1)/bert_mask[:,1:].sum(1).unsqueeze(-1)
        if (self.test_mode):
            bert_output_mean_pooling = bert_output_mean_pooling.half()
        else:
            bert_output_mean_pooling = bert_output_mean_pooling.float()
            
        bert_output_max_pooling = (bert_output_last_hidden_state[:,1:,:]+(1-bert_mask[:,1:]).unsqueeze(-1)*(-1e10)).max(1)[0]
        if (self.test_mode):
            bert_output_max_pooling = bert_output_max_pooling.half()
        else:
            bert_output_max_pooling = bert_output_max_pooling.float()
        
        text_output_mean_pooling = (bert_output_last_hidden_state[:,(1+self.max_frames):,:]*bert_mask[:,(1+self.max_frames):].unsqueeze(-1)).sum(1)/bert_mask[:,(1+self.max_frames):].sum(1).unsqueeze(-1)
        if (self.test_mode):
            text_output_mean_pooling = text_output_mean_pooling.half()
        else:
            text_output_mean_pooling = text_output_mean_pooling.float()
            
        text_output_max_pooling = (bert_output_last_hidden_state[:,(1+self.max_frames):,:]+(1-bert_mask[:,(1+self.max_frames):]).unsqueeze(-1)*(-1e10)).max(1)[0]
        if (self.test_mode):
            text_output_max_pooling = text_output_max_pooling.half()
        else:
            text_output_max_pooling = text_output_max_pooling.float()
        
        vision_output_mean_pooling = (bert_output_last_hidden_state[:,1:(1+self.max_frames),:]*bert_mask[:,1:(1+self.max_frames)].unsqueeze(-1)).sum(1)/bert_mask[:,1:(1+self.max_frames)].sum(1).unsqueeze(-1)
        if (self.test_mode):
            vision_output_mean_pooling = vision_output_mean_pooling.half()
        else:
            vision_output_mean_pooling = vision_output_mean_pooling.float()
        
        vision_output_max_pooling = (bert_output_last_hidden_state[:,1:(1+self.max_frames),:]+(1-bert_mask[:,1:(1+self.max_frames)]).unsqueeze(-1)*(-1e10)).max(1)[0]
        if (self.test_mode):        
            vision_output_max_pooling = vision_output_max_pooling.half()
        else:
            vision_output_max_pooling = vision_output_max_pooling.float()
            
        vision_vlad_embedding = self.nextvlad(vision_input_embedding, vision_input_mask)
        vision_vlad_embedding = self.enhance(vision_vlad_embedding)

        final_embedding = torch.cat([bert_output_cls, bert_output_mean_pooling, bert_output_max_pooling, text_output_mean_pooling, text_output_max_pooling, vision_output_mean_pooling, vision_output_max_pooling, vision_vlad_embedding], dim=1)
        
        # final_embedding = torch.cat([bert_output_cls, bert_output_mean_pooling, bert_output_max_pooling, text_output_mean_pooling, text_output_max_pooling, vision_output_mean_pooling, vision_output_max_pooling], dim=1)
        
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1), F.softmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'].cuda())

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label).cuda()
        with torch.no_grad():
            pred_label_id = torch.argmax(prediction, dim=1)
            accuracy = (label == pred_label_id).float().sum() / label.shape[0]
        return loss, accuracy, pred_label_id, label


class NeXtVLAD(nn.Module):
    def __init__(self, feature_size, cluster_size, output_size=1024, expansion=2, groups=8, dropout=0.2):
        super().__init__()
        self.feature_size = feature_size
        self.output_size = output_size
        self.expansion_size = expansion
        self.cluster_size = cluster_size
        self.groups = groups
        self.drop_rate = dropout

        self.new_feature_size = self.expansion_size * self.feature_size // self.groups

        self.dropout = torch.nn.Dropout(self.drop_rate)
        self.expansion_linear = torch.nn.Linear(self.feature_size, self.expansion_size * self.feature_size)
        self.group_attention = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups)
        self.cluster_linear = torch.nn.Linear(self.expansion_size * self.feature_size, self.groups * self.cluster_size,
                                              bias=False)
        self.cluster_weight = torch.nn.Parameter(
            torch.nn.init.normal_(torch.rand(1, self.new_feature_size, self.cluster_size), std=0.01))
        self.fc = torch.nn.Linear(self.new_feature_size * self.cluster_size, self.output_size)

    def forward(self, inputs, mask):
        # todo mask
        inputs = self.expansion_linear(inputs)
        attention = self.group_attention(inputs)
        attention = torch.sigmoid(attention)
        attention = attention.reshape([-1, inputs.size(1) * self.groups, 1])
        reshaped_input = inputs.reshape([-1, self.expansion_size * self.feature_size])
        activation = self.cluster_linear(reshaped_input)
        activation = activation.reshape([-1, inputs.size(1) * self.groups, self.cluster_size])
        activation = torch.softmax(activation, dim=-1)
        activation = activation * attention
        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weight
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = inputs.reshape([-1, inputs.shape[1] * self.groups, self.new_feature_size])
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 2, 1).contiguous()
        vlad = F.normalize(vlad - a, p=2, dim=1)
        vlad = vlad.reshape([-1, self.cluster_size * self.new_feature_size])
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


# class ConcatDenseSE(nn.Module):
#     def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
#         super().__init__()
#         self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
#         self.fusion_dropout = nn.Dropout(dropout)
#         self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

#     def forward(self, inputs):
#         embeddings = torch.cat(inputs, dim=1)
#         embeddings = self.fusion_dropout(embeddings)
#         embedding = self.fusion(embeddings)
#         embedding = self.enhance(embedding)

#         return embedding

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
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)