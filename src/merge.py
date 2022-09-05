import numpy as np
import pandas as pd

import torch
from category_id_map import lv2id_to_category_id

result_id_list = pd.read_csv('save/finetune_mlm_mfm_itm_base/result_label.csv', header = None, dtype = str)[0].values

file_list_0 = [
             'save/finetune_mlm_mfm_itm_base/result_prob_0.npy',
             'save/finetune_mlm_mfm_itm_short_seq/result_prob_0.npy',
             'save/finetune_mlm_mfm_itm_long_seq/result_prob_0.npy',
             'save/finetune_mlm_itm_short_seq/result_prob_0.npy',
             'save/finetune_mlm_itm_long_seq/result_prob_0.npy']

file_list_1 = [
             'save/finetune_mlm_mfm_itm_base/result_prob_1.npy',
             'save/finetune_mlm_mfm_itm_short_seq/result_prob_1.npy',
             'save/finetune_mlm_mfm_itm_long_seq/result_prob_1.npy',
             'save/finetune_mlm_itm_short_seq/result_prob_1.npy',
             'save/finetune_mlm_itm_long_seq/result_prob_1.npy']

avg_result = np.zeros((len(result_id_list),200))
weights = [1, 1, 1, 1, 1]

for i in range(0, len(file_list_0)):
    avg_result[:len(result_id_list)//2, :] += weights[i] * np.load(open(file_list_0[i], 'rb'))
    avg_result[len(result_id_list)//2:, :] += weights[i] * np.load(open(file_list_1[i], 'rb'))
    
final_label_list = np.argmax(avg_result, axis=1)

with open('/opt/ml/output/result.csv', 'w+') as f:
    for result_id, final_label in zip(result_id_list, final_label_list):
        category_id = lv2id_to_category_id(final_label)
        f.write(f'{result_id},{category_id}\n')