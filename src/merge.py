import pandas as pd
import numpy as np

import torch
from category_id_map import lv2id_to_category_id

result_id_list = pd.read_csv('data/results/vlbert_base_mlm_mfm_itm/result_label.csv', header = None, dtype = str)[0].values

file_list_1 = [
             'data/results/vlbert_base_mlm_mfm_itm_fold_0/result_prob.csv',
             'data/results/vlbert_base_mlm_mfm_itm_fold_1/result_prob.csv',
             'data/results/vlbert_base_mlm_mfm_itm_fold_2/result_prob.csv',
             'data/results/vlbert_base_mlm_mfm_itm_fold_3/result_prob.csv',
             'data/results/vlbert_base_mlm_mfm_itm_fold_4/result_prob.csv']

file_list_2 = [
             'data/results/vlbert_base_mlm_itm/result_prob.csv',
             'data/results/vlbert_base_mlm_mfm_itm/result_prob.csv',
             'data/results/vlbert_large_mlm_mfm_itm/result_prob.csv',
             'data/results/albef_mlm_itm/result_prob.csv']

avg_result = np.zeros((25000,200))

for file in file_list_1:
    avg_result += 0.2*pd.read_csv(file, header = None).values
for file in file_list_2:
    avg_result += pd.read_csv(file, header = None).values
    
final_label_list = np.argmax(avg_result, axis=1)

with open(f'data/result.csv', 'w+') as f:
    for result_id, final_label in zip(result_id_list, final_label_list):
        category_id = lv2id_to_category_id(final_label)
        f.write(f'{result_id},{category_id}\n')