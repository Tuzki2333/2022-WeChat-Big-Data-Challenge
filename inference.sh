cd src/

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_itm/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_itm'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_mfm_itm/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_mfm_itm'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_mfm_itm_fold_0/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_mfm_itm_fold_0'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_mfm_itm_fold_1/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_mfm_itm_fold_1'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_mfm_itm_fold_2/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_mfm_itm_fold_2'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_mfm_itm_fold_3/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_mfm_itm_fold_3'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_base_mlm_mfm_itm_fold_4/model_step_9000.bin' --test_output_path 'vlbert_base_mlm_mfm_itm_fold_4'

python vlbert_finetune_inference.py --ckpt_file 'data/finetune_save/vlbert_large_mlm_mfm_itm/model_step_9000.bin' --bert_output_size 1024 --bert_freezing_layers 12 --bert_dir 'hfl/chinese-roberta-wwm-ext-large' --test_output_path 'vlbert_large_mlm_mfm_itm'

python albef_finetune_inference.py --ckpt_file 'data/finetune_save/albef_mlm_itm/model_step_9000.bin' --test_output_path 'albef_mlm_itm'

python merge.py