cd /opt/ml/wxcode/src

python vlbert_pretrain_main.py --use_mfm True --model_name pretrain_mlm_mfm_itm --save_steps 110000

python vlbert_pretrain_main.py --model_name pretrain_mlm_itm --save_steps 100000

python vlbert_finetune_main.py \
    --end_to_end_mode True --max_frames 16 \
    --use_ema True \
    --use_fgm True \
    --default_learning_rate 1e-4 \
    --bert_learning_rate 2e-5 \
    --swin_learning_rate 3e-5 \
    --pretrain_ckpt_file save/pretrain_mlm_mfm_itm/model_step_110000.bin \
    --model_name finetune_mlm_mfm_itm_base
    
python extract_feature_train.py \
    --max_frames 32 \
    --zip_frame_dir /opt/ml/input/data/zip_frames/labeled/ \
    --ann_path /opt/ml/input/data/annotations/labeled.json \
    --model_pretrained_path finetune_mlm_mfm_itm_base/model_step_19000.bin \
    --output_path save/zip_feats/labeled_vlbert.zip
    
python vlbert_finetune_main.py \
    --seed 1 \
    --max_frames 16 \
    --bert_seq_length 256 \
    --use_ema True \
    --use_fgm True \
    --default_learning_rate 1e-4 \
    --bert_learning_rate 2e-5 \
    --train_zip_feat_path save/zip_feats/labeled_vlbert.zip \
    --pretrain_ckpt_file save/pretrain_mlm_itm/model_step_100000.bin \
    --model_name finetune_mlm_itm_short_seq

python vlbert_finetune_main.py \
    --seed 2 \
    --max_frames 16 \
    --bert_seq_length 384 \
    --use_ema True \
    --use_fgm True \
    --default_learning_rate 1e-4 \
    --bert_learning_rate 2e-5 \
    --train_zip_feat_path save/zip_feats/labeled_vlbert.zip \
    --pretrain_ckpt_file save/pretrain_mlm_itm/model_step_100000.bin \
    --model_name finetune_mlm_itm_long_seq
    
python vlbert_finetune_main.py \
    --seed 3 \
    --max_frames 16 \
    --bert_seq_length 256 \
    --use_ema True \
    --use_fgm True \
    --default_learning_rate 1e-4 \
    --bert_learning_rate 2e-5 \
    --train_zip_feat_path save/zip_feats/labeled_vlbert.zip \
    --pretrain_ckpt_file save/pretrain_mlm_mfm_itm/model_step_110000.bin \
    --model_name finetune_mlm_mfm_itm_short_seq
    
python vlbert_finetune_main.py \
    --seed 4 \
    --max_frames 16 \
    --bert_seq_length 384 \
    --use_ema True \
    --use_fgm True \
    --default_learning_rate 1e-4 \
    --bert_learning_rate 2e-5 \
    --train_zip_feat_path save/zip_feats/labeled_vlbert.zip \
    --pretrain_ckpt_file save/pretrain_mlm_mfm_itm/model_step_110000.bin \
    --model_name finetune_mlm_mfm_itm_long_seq