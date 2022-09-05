## [2022中国高校计算机大赛-微信大数据挑战赛](https://algo.weixin.qq.com/) 复赛代码提交

### 开源模型

* 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext` 模型作为预训练的初始权重。链接为：https://huggingface.co/hfl/chinese-roberta-wwm-ext
* 使用了 Swin-Transformer 模型作为视觉特征抽取模型。链接为：https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth

### 代码介绍

大部分核心代码、函数与初赛保持一致：

- category_id_map.py 是category_id 和一级、二级分类的映射
- vlbert_pretrain_config.py 是预训练的配置文件
- vlbert_finetune_config.py 是微调的配置文件
- pretrain_data_helper.py 是预训练的数据预处理模块
- finetune_data_helper.py 是微调的数据预处理模块
- vlbert_finetune_inference.py 生成提交文件
- vlbert_pretrain_main.py 是模型预训练的入口
- vlbert_finetune_main.py 是模型微调的入口
- vlbert_pretrain_model.py 是预训练模型
- vlbert_finetune_model.py 是微调模型
- util.py 是util函数
- merge.py 模型融合函数

区别在于，复赛不再提供预提取好的视觉特征，因此需要在训练、测试中自行提取视觉特征。在代码中，增加了如下文件：

- extract_feature_train.py 训练阶段提取视觉特征的函数
- extract_feature_inference.py 推理阶段提取视觉特征的函数

此外，我们用到了一些开源代码，包括：

- third_party/ema.py 出自https://www.cnblogs.com/sddai/p/14646581.html
- third_party/fgm.py 出自https://blog.csdn.net/znevegiveup1/article/details/121430605
- third_party/qq_pretrain.py 出自 https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st
- third_party/swin2.py 出自https://github.com/SwinTransformer


### 模型描述

* 视频特征：利用swin v2进行抽取，相比v1模型有所提升
* 文本特征：采用[SEP] + title + [SEP] + asr + [SEP] +  ocr的拼接方式，其中，若文本超出允许的最大长度，则优先截断ocr，其次截断asr，以此类推；当截断ocr时，只保留ocr的后半部分信息
* vlbert模型：视频特征经过embedding之后和文本特征拼接，一起输入bert的encoder中
* 模型输出：提取encoder的（1）last hidden state的cls、（2）文本部分mean pooling、（3）文本部分max pooling、（4）视频部分mean pooling、（5）视频部分max pooling、（6）文本+视频部分mean pooling、（7）文本+视频部分max pooling，拼接在一起之后，和nextvlad的输出进行拼接，最后经过一个mlp，得到最终分类

### 训练流程

#### 预训练

分别执行两次预训练。一是在base模型上做mlm和itm的预训练，二是在base模型上做mlm、mfm和itm的预训练。每次预训练的执行时间约26小时。

#### 微调

* 首先，对mlm + mfm + itm预训练后的模型进行端到端的微调（swin也参与微调），微调过程中引入ema和fgm，采用全量数据，执行时间约12小时。
* 利用第一次微调得出的swin模型，提取视觉特征，供后续模型使用。
* 对mlm + mfm + itm预训练后的模型进行非端到端的微调（swin不参与微调，采用第一个模型微调出的参数），文本长度分别取256和384，得到两个模型。
* 对mlm + itm预训练后的模型进行非端到端的微调（swin不参与微调，采用第一个模型微调出的参数），文本长度分别取256和384，得到两个模型。
* 至此，一共得到五个模型。

#### 模型融合

按照输出概率进行等权重融合

### 性能

#### B榜测试性能

0.702343

### 主要贡献点和核心代码片段

* 文本输入方面，尝试了不同的截断方式，最后采用了保留ocr后半部分信息这一截断方式。

```python
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
```

* 模型结构方面，根据多模态的特点，对vlbert模型的输出提取了（1）last hidden state的cls、（2）文本部分mean pooling、（3）文本部分max pooling、（4）视频部分mean pooling、（5）视频部分max pooling、（6）文本+视频部分mean pooling、（7）文本+视频部分max pooling。有效提升了效果。

```python
bert_output_cls = self.bert.pooler(bert_output_last_hidden_state)

bert_output_mean_pooling = (bert_output_last_hidden_state[:,1:,:]*bert_mask[:,1:].unsqueeze(-1)).sum(1)/bert_mask[:,1:].sum(1).unsqueeze(-1)
bert_output_max_pooling = (bert_output_last_hidden_state[:,1:,:]+(1-bert_mask[:,1:]).unsqueeze(-1)*(-1e10)).max(1)[0]

text_output_mean_pooling = (bert_output_last_hidden_state[:,(1+self.max_frames):,:]*bert_mask[:,(1+self.max_frames):].unsqueeze(-1)).sum(1)/bert_mask[:,(1+self.max_frames):].sum(1).unsqueeze(-1)
text_output_max_pooling = (bert_output_last_hidden_state[:,(1+self.max_frames):,:]+(1-bert_mask[:,(1+self.max_frames):]).unsqueeze(-1)*(-1e10)).max(1)[0]

vision_output_mean_pooling = (bert_output_last_hidden_state[:,1:(1+self.max_frames),:]*bert_mask[:,1:(1+self.max_frames)].unsqueeze(-1)).sum(1)/bert_mask[:,1:(1+self.max_frames)].sum(1).unsqueeze(-1)
vision_output_max_pooling = (bert_output_last_hidden_state[:,1:(1+self.max_frames),:]+(1-bert_mask[:,1:(1+self.max_frames)]).unsqueeze(-1)*(-1e10)).max(1)[0]
```

* 将单流模型的输出和nextvlad进行拼接。

```python
final_embedding = torch.cat([bert_output_cls, bert_output_mean_pooling, bert_output_max_pooling, text_output_mean_pooling, text_output_max_pooling, vision_output_mean_pooling, vision_output_max_pooling, vision_vlad_embedding], dim=1)
```

* 训练方式方面，采用一次端到端微调 + 数次非端到端微调的方式。这样一来，尽管我们改变了swin模型的参数，但也能有效地进行模型融合，使得测试结果进一步提升。