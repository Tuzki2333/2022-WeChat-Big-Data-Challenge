import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from transformers.models.bert.modeling_bert import BertConfig, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder
from transformers import BertTokenizer

from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class MaskWord(object):
    def __init__(self, tokenizer_path, cache_dir, mlm_probability=0.15, config = None):
        self.mlm_probability = 0.15
        if (config):
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, use_fast=True, cache_dir=cache_dir)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, config=config, use_fast=True, cache_dir=cache_dir)
        
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
class MaskVideo(object):
    def __init__(self, mlm_probability=0.15):
        self.mlm_probability = 0.15
        
    def torch_mask_frames(self, video_feature, video_mask):
        probability_matrix = torch.full(video_mask.shape, 0.9 * self.mlm_probability)
        probability_matrix = probability_matrix * video_mask
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        video_labels_index = torch.arange(video_feature.size(0) * video_feature.size(1)).view(-1, video_feature.size(1))
        video_labels_index = -100 * ~masked_indices + video_labels_index * masked_indices

        # 90% mask video fill all 0.0
        masked_indices_unsqueeze = masked_indices.unsqueeze(-1).expand_as(video_feature)
        inputs = video_feature.data.masked_fill(masked_indices_unsqueeze, 0.0)
        labels = video_feature[masked_indices_unsqueeze].contiguous().view(-1, video_feature.size(2)) 
        return inputs, video_mask, video_labels_index
    
class ShuffleVideo(object):
    def __init__(self):
        pass
    
    def torch_shuf_video(self, video_feature):
        bs = video_feature.size()[0]
        # batch 内前一半 video 保持原顺序，后一半 video 逆序
        shuf_index = torch.tensor(list(range(bs // 2)) + list(range(bs //2, bs))[::-1])
        # shuf 后的 label
        label = (torch.tensor(list(range(bs))) == shuf_index).float()
        video_feature = video_feature[shuf_index]
        # video_mask = video_mask[shuf_index]
        return video_feature, label
    
class VisualPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class VisualLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = VisualPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, 768, bias=False)
        self.bias = nn.Parameter(torch.zeros(768))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class VisualOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = VisualLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    
    
# calc mfm loss 
def calculate_mfm_loss(video_feature_output, video_feature_input, 
                       video_mask, video_labels_index, normalize=False, temp=0.1):

    if normalize:
        video_feature_output = torch.nn.functional.normalize(video_feature_output, p=2, dim=2)
        video_feature_input = torch.nn.functional.normalize(video_feature_input, p=2, dim=2)

    afm_scores_tr = video_feature_output.view(-1, video_feature_output.shape[-1])

    video_tr = video_feature_input.permute(2, 0, 1)
    video_tr = video_tr.view(video_tr.shape[0], -1)

    logits_matrix = torch.mm(afm_scores_tr, video_tr)
    if normalize:
        logits_matrix = logits_matrix / temp

    video_mask_float = video_mask.to(dtype=torch.float)
    mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
    masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

    logpt = F.log_softmax(masked_logits, dim=-1)
    logpt = torch.diag(logpt)
    nce_loss = -logpt

    video_labels_index_mask = (video_labels_index != -100)
    nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
    nce_loss = nce_loss.mean()
    return nce_loss