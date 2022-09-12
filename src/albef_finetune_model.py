import torch
from torch import nn
import torch.nn.functional as F

from third_party.xbert import BertConfig, BertModel, BertEmbeddings
import transformers
# from transformers.models.bert.modeling_bert import BertConfig
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder

from category_id_map import CATEGORY_ID_LIST, LV1ID, lv2id_to_lv1id, lv1id_to_nest

class ALBEF(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.use_vlad = args.use_vlad
        
        self.bert_config = BertConfig.from_json_file("third_party/bert.json")
        self.vision_config = transformers.BertConfig(num_hidden_layers=args.vision_layer_num, vocab_size=21128)
        self.visual_encoder = transformers.BertModel.from_pretrained(args.bert_dir, config=self.vision_config)
        self.visual_encoder = randomize_model(self.visual_encoder)
        
        self.bert = BertModel.from_pretrained(args.bert_dir, config=self.bert_config, add_pooling_layer=False)

        if (self.use_vlad):
            self.nextvlad = NeXtVLAD(args.frame_embedding_size, args.vlad_cluster_size,
                                     output_size=args.vlad_hidden_size, dropout=args.dropout)
            self.enhance = SENet(channels=args.vlad_hidden_size, ratio=args.se_ratio)
            vlad_output_size = args.vlad_hidden_size
        else:
            vlad_output_size = 0

        self.classifier = MultiLayerPerceptron(
            3 * args.bert_output_size + vlad_output_size, len(CATEGORY_ID_LIST),
            embed_sizes=args.classifier_mlp_sizes, last_layer_activation=False, dropout=args.dropout)

        self.device = args.device
        self.vision_fc = MultiLayerPerceptron(args.frame_embedding_size, args.bert_output_size, bn = False)

    def forward(self, inputs, inference=False):
        embedding_all = []
        
        text_input_mask = inputs['concat_text_mask']
        # vision_input = self.vision_fc(inputs['frame_input'])
        
        vision_input = inputs['frame_input']
        image_embeds = self.visual_encoder.encoder(vision_input, attention_mask = (1.0 -  inputs['frame_mask'][:,None,None,:]) * -10000.0)['last_hidden_state']
        

        output = self.bert(inputs['concat_text_input'],
                                   attention_mask= text_input_mask,
                                   encoder_hidden_states=image_embeds,
                                   encoder_attention_mask= inputs['frame_mask'],
                                   return_dict=True
                                   )
        bert_output_last_hidden_state = output.last_hidden_state
        bert_output_cls = bert_output_last_hidden_state[:, 0, :]

        bert_output_mean_pooling = (bert_output_last_hidden_state[:, 1:, :] * text_input_mask[:, 1:].unsqueeze(-1)).sum(1) / text_input_mask[:, 1:].sum(1).unsqueeze(-1)
        bert_output_mean_pooling = bert_output_mean_pooling.float()
        bert_output_max_pooling = (bert_output_last_hidden_state[:, 1:, :] + (1 - text_input_mask[:, 1:]).unsqueeze(-1) * (-1e10)).max(1)[0]
        bert_output_max_pooling = bert_output_max_pooling.float()
        bert_output_embedding = torch.cat((bert_output_cls, bert_output_mean_pooling, bert_output_max_pooling), -1)
        embedding_all.append(bert_output_embedding)

        if (self.use_vlad):
            vision_embedding = self.nextvlad(inputs['frame_input'], inputs['frame_mask'])
            vision_embedding = self.enhance(vision_embedding)
            embedding_all.append(vision_embedding)

        final_embedding = torch.cat(embedding_all, dim=1)

        # final_embedding = self.fusion(embedding_all)
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