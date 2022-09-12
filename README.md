## 代码说明

### 硬件配置

我们的代码涉及到预训练和微调，且涉及到base和large模型，因此不同的执行过程采用的硬件配置有所不同：
* base模型上的预训练：要求单卡，且显存不小于80G（在单卡A100中测试通过）
* large模型上的预训练：要求多卡，显存不小于240G（在三卡A100中测试通过）
* base模型上的微调：要求单卡，且显存不小于24G（在单卡A5000中测试通过）
* large模型上的微调：要求单卡，且显存不小于80G（在单卡A100中测试通过）

以上用单卡执行的任务并没有在多卡环境中进行调整和测试，为了防止报错，我们在提交代码时强行只设置了一个GPU可见。如果单卡显存足够的话，无论是多卡配置还是单卡配置，都应当可以顺利运行

### 环境配置

* Python 版本：3.8
* PyTorch 版本：1.10.0
* CUDA 版本：11.3

所需环境在 `requirements.txt` 中定义。

### 数据

* 使用大赛提供的有标注数据（10万）和无标注数据（100万）。
* 未使用任何额外数据。

### 开源模型

* 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext` 模型作为预训练的初始权重。链接为：https://huggingface.co/hfl/chinese-roberta-wwm-ext
* 使用了 huggingface 上提供的 `hfl/chinese-roberta-wwm-ext-large` 模型作为预训练的初始权重。链接为： https://huggingface.co/hfl/chinese-roberta-wwm-ext-large

### 模型描述

* 文本特征：采用[SEP] + title + [SEP] + asr + [SEP] +  ocr的拼接方式，其中，若文本超出允许的最大长度，则优先截断ocr，其次截断asr，以此类推；当截断ocr时，只保留ocr的后半部分信息
* 视频特征：和baseline保持一致
* vlbert模型：视频特征经过embedding之后和文本特征拼接，一起输入bert的encoder中
* albef模型：视频特征和文本特征都经过单独的encoder层之后，输入cross transformer中进行交互
* 模型输出：提取encoder的last hidden state的cls、mean pooling和max pooling信息，拼接在一起之后，和nextvlad的输出进行拼接，最后经过一个mlp，得到最终分类

### 训练流程

#### 预训练

* vlbert模型：分别执行三次预训练。一是在base模型上做mlm和itm的预训练，二是在base模型上做mlm、mfm和itm的预训练，三是在large模型上做mlm、mfm和itm的预训练
* albef模型：执行一次预训练，任务为mlm和itm
* 除了large模型上的预训练外，其它的预训练任务均可在单卡A100（显存80G）上完成
* large模型的预训练任务在三卡A100上完成

#### 微调

* vlbert_base + mlm + itm：采用单折数据进行训练（skf划分90%的数据为训练集），使用的trick包括ema和fgm，一共微调出1个模型
* vlbert_base + mlm + mfm + itm：采用全量和五折数据进行训练，使用的trick包括ema和fgm，一共微调出6个模型
* vlbert_large + mlm + mfm + itm：采用单折数据进行训练，为了减缓过拟合额外冻结12层bert encoder，使用的trick包括fgm，一共微调出1个模型
* albef_base + mlm + itm：采用单折数据进行训练，使用的trick包括ema和fgm，一共微调出1个模型
* 除了large模型上的微调外，其它的微调任务均可在单卡A5000（显存24G）上完成
* large模型的微调任务在单卡A100（显存80G）上完成

#### 模型融合

按照输出概率进行加权融合，其中vlbert_base + mlm + mfm + itm中的五折结果赋权重0.2，其它结果权重为1

### 代码结构

主要是基于baseline的代码结构进行了修改，分为四大部分（vlbert-pretrain, vlbert-finetune, albef-pretrain, albef-finetune)
主要使用的第三方开源代码包括：
* EMA：https://www.cnblogs.com/sddai/p/14646581.html
* FGM：https://blog.csdn.net/znevegiveup1/article/details/121430605
* QQ浏览器2021AI算法大赛赛道一第1名方案：https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st
* ALBEF：https://github.com/salesforce/ALBEF

### 性能

#### 离线测试性能

* vlbert_base + mlm + itm：单折0.6898
* vlbert_base + mlm + mfm + itm：五折分别是0.6911、0.6908、0.6872、0.6905、0.6877
* vlbert_large + mlm + mfm + itm：单折0.7017
* albef_base + mlm + itm：单折0.692

#### B榜测试性能

0.694421