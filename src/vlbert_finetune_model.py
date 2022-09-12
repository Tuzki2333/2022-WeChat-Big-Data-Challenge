import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

from category_id_map import CATEGORY_ID_LIST, LV1ID, lv2id_to_lv1id, lv1id_to_nest


class MultiModal(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.device = args.device
        self.use_vlad = args.use_vlad
        self.use_vision_bert_emb = args.use_vision_bert_emb
        self.bert_output_size = args.bert_output_size
        self.bert = BertModel.from_pretrained(args.bert_dir, cache_dir = args.bert_cache, output_hidden_states=True, output_attentions=True)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir, cache_dir = args.bert_cache)
        
        if (self.use_vlad):
            self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                     output_size=args.vlad_hidden_size, dropout=args.dropout)
            self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
            vlad_output_size = args.vlad_hidden_size
        else:
            vlad_output_size = 0
        
        self.vision_fc = MultiLayerPerceptron(args.frame_embedding_size, args.bert_output_size, bn = False)
        self.vision_bert_embeddings = BertEmbeddings(self.bert_config)
        
        self.classifier = MultiLayerPerceptron(7*args.bert_output_size + vlad_output_size, len(CATEGORY_ID_LIST), embed_sizes = args.classifier_mlp_sizes, last_layer_activation = False, dropout = args.dropout)

    def forward(self, inputs, inference=False):
        
        embedding_all = []
        
        text_input_embedding = self.bert.embeddings(input_ids = inputs['concat_text_input'])
        cls_input_embedding = text_input_embedding[:, 0:1, :]
        text_input_embedding = text_input_embedding[:, 1:, :]
        
        text_input_mask = inputs['concat_text_mask']
        cls_input_mask = text_input_mask[:, 0:1]
        text_input_mask = text_input_mask[:, 1:]
            
        vision_input_embedding = self.vision_fc(inputs['frame_input'])
        
        if (self.use_vision_bert_emb):
            vision_input_embedding = self.vision_bert_embeddings(inputs_embeds = vision_input_embedding)

        bert_input_embedding = torch.cat((cls_input_embedding, vision_input_embedding, text_input_embedding), dim = 1)
        bert_mask  = torch.cat((cls_input_mask, inputs['frame_mask'], text_input_mask), dim = 1)
        
        bert_output = self.bert.encoder(bert_input_embedding, attention_mask = (1.0 - bert_mask[:,None,None,:]) * -10000.0, output_hidden_states=True)
        
        bert_output_last_hidden_state = bert_output['last_hidden_state']
        bert_output_cls = self.bert.pooler(bert_output_last_hidden_state)
        
        bert_output_mean_pooling = (bert_output_last_hidden_state[:,1:,:]*bert_mask[:,1:].unsqueeze(-1)).sum(1)/bert_mask[:,1:].sum(1).unsqueeze(-1)
        bert_output_mean_pooling = bert_output_mean_pooling.float()
        bert_output_max_pooling = (bert_output_last_hidden_state[:,1:,:]+(1-bert_mask[:,1:]).unsqueeze(-1)*(-1e10)).max(1)[0]
        bert_output_max_pooling = bert_output_max_pooling.float()
        
        text_output_mean_pooling = (bert_output_last_hidden_state[:,33:,:]*bert_mask[:,33:].unsqueeze(-1)).sum(1)/bert_mask[:,33:].sum(1).unsqueeze(-1)
        text_output_mean_pooling = text_output_mean_pooling.float()
        text_output_max_pooling = (bert_output_last_hidden_state[:,33:,:]+(1-bert_mask[:,33:]).unsqueeze(-1)*(-1e10)).max(1)[0]
        text_output_max_pooling = text_output_max_pooling.float()
        
        vision_output_mean_pooling = (bert_output_last_hidden_state[:,1:33,:]*bert_mask[:,1:33].unsqueeze(-1)).sum(1)/bert_mask[:,1:33].sum(1).unsqueeze(-1)
        vision_output_mean_pooling = vision_output_mean_pooling.float()
        vision_output_max_pooling = (bert_output_last_hidden_state[:,1:33,:]+(1-bert_mask[:,1:33]).unsqueeze(-1)*(-1e10)).max(1)[0]
        vision_output_max_pooling = vision_output_max_pooling.float()
        
        bert_output_embedding = torch.cat((bert_output_cls, bert_output_mean_pooling, text_output_mean_pooling, vision_output_mean_pooling, bert_output_max_pooling, text_output_max_pooling, vision_output_max_pooling),-1)
        
        embedding_all.append(bert_output_embedding)
            
        if (self.use_vlad):
            vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
            vision_embedding = self.enhance(vision_embedding)
            embedding_all.append(vision_embedding)
        
        final_embedding = torch.cat(embedding_all, dim=1)
        
        prediction = self.classifier(final_embedding)

        if inference:
            return torch.argmax(prediction, dim=1), F.softmax(prediction, dim=1)
        else:
            return self.cal_loss(prediction, inputs['label'])

    @staticmethod
    def cal_loss(prediction, label):
        label = label.squeeze(dim=1)
        loss = F.cross_entropy(prediction, label)
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
        self.bn = torch.nn.BatchNorm1d(self.new_feature_size * self.cluster_size)
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
        vlad = self.bn(vlad)
        vlad = self.dropout(vlad)
        vlad = self.fc(vlad)
        return vlad


class SENet(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.sequeeze = nn.Linear(in_features=channels, out_features=channels // ratio, bias=False)
        self.relu = nn.ReLU()
        # self.relu = gelu()
        self.excitation = nn.Linear(in_features=channels // ratio, out_features=channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        gates = self.sequeeze(x)
        gates = self.relu(gates)
        gates = self.excitation(gates)
        gates = self.sigmoid(gates)
        x = torch.mul(x, gates)

        return x


class ConcatDenseSE(nn.Module):
    def __init__(self, multimodal_hidden_size, hidden_size, se_ratio, dropout):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(multimodal_hidden_size)
        self.fusion = nn.Linear(multimodal_hidden_size, hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)
        self.enhance = SENet(channels=hidden_size, ratio=se_ratio)

    def forward(self, inputs):
        embeddings = torch.cat(inputs, dim=1)
        embeddings = self.bn(embeddings)
        embeddings = self.fusion_dropout(embeddings)
        embedding = self.fusion(embeddings)
        embedding = self.enhance(embedding)

        return embedding
    
    
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