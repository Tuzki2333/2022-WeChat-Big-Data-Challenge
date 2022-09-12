cd src/

python vlbert_pretrain_main.py --savedmodel_path 'vlbert_large_mlm_mfm_itm' --bert_dir 'hfl/chinese-roberta-wwm-ext-large' --bert_seq_length_concat 256 --bert_output_size 1024 --batch_size 210 --use_mfm True --use_parallel True --save_steps 50000

python vlbert_pretrain_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm' --batch_size 108 --use_mfm True --save_steps 100000

python vlbert_pretrain_main.py --savedmodel_path 'vlbert_base_mlm_itm' --batch_size 108 --save_steps 110000

python albef_pretrain_main.py --savedmodel_path 'albef_mlm_itm' --batch_size 110 --save_steps 100000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_large_mlm_mfm_itm' --pretrain_ckpt_file 'data/pretrain_save/vlbert_large_mlm_mfm_itm/model_step_50000.bin' --bert_dir 'hfl/chinese-roberta-wwm-ext-large' --bert_output_size 1024 --bert_freezing_layers 12 --bert_layerwise_learning_rate_decay 1.0 --use_fgm True --use_val True --n_splits 10 --fold 0 --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_itm' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_itm/model_step_110000.bin' --use_ema True --use_fgm True --use_val True --n_splits 10 --fold 0 --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_mfm_itm/model_step_100000.bin' --use_ema True --use_fgm True --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm_fold_0' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_mfm_itm/model_step_100000.bin' --use_ema True --use_fgm True --use_val True --n_splits 5 --fold 0 --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm_fold_1' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_mfm_itm/model_step_100000.bin' --use_ema True --use_fgm True --use_val True --n_splits 5 --fold 1 --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm_fold_2' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_mfm_itm/model_step_100000.bin' --use_ema True --use_fgm True --use_val True --n_splits 5 --fold 2 --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm_fold_3' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_mfm_itm/model_step_100000.bin' --use_ema True --use_fgm True --use_val True --n_splits 5 --fold 3 --save_steps 9000

python vlbert_finetune_main.py --savedmodel_path 'vlbert_base_mlm_mfm_itm_fold_4' --pretrain_ckpt_file 'data/pretrain_save/vlbert_base_mlm_mfm_itm/model_step_100000.bin' --use_ema True --use_fgm True --use_val True --n_splits 5 --fold 4 --save_steps 9000

python albef_finetune_main.py --savedmodel_path 'albef_mlm_itm' --pretrain_ckpt_file 'data/pretrain_save/albef_mlm_itm/model_step_100000.bin' --use_ema True --use_fgm True --use_val True --save_steps 9000