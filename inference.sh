cd /opt/ml/wxcode/src

python -u -m torch.distributed.launch --nproc_per_node=2 extract_feature_inference.py \
    --max_frames 16 \
    --model_pretrained_path save/finetune_mlm_mfm_itm_base/model_step_19000.bin \
    --ann_path /opt/ml/input/data/annotations/test.json \
    --zip_frame_dir /opt/ml/input/data/zip_frames/test/ \
    --output_path temp

python -u -m torch.distributed.launch --nproc_per_node=2 vlbert_finetune_inference.py \
    --max_frames 16 --bert_seq_length 256 \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --test_zip_feat_path temp \
    --ckpt_file save/finetune_mlm_mfm_itm_base/model_step_19000.bin \
    --test_output_path save/finetune_mlm_mfm_itm_base/

python -u -m torch.distributed.launch --nproc_per_node=2 vlbert_finetune_inference.py \
    --max_frames 16 --bert_seq_length 256 \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --test_zip_feat_path temp \
    --ckpt_file save/finetune_mlm_mfm_itm_short_seq/model_step_19000.bin \
    --test_output_path save/finetune_mlm_mfm_itm_short_seq/

python -u -m torch.distributed.launch --nproc_per_node=2 vlbert_finetune_inference.py \
    --max_frames 16 --bert_seq_length 384 \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --test_zip_feat_path temp \
    --ckpt_file save/finetune_mlm_mfm_itm_long_seq/model_step_19000.bin \
    --test_output_path save/finetune_mlm_mfm_itm_long_seq/
    
python -u -m torch.distributed.launch --nproc_per_node=2 vlbert_finetune_inference.py \
    --max_frames 16 --bert_seq_length 256 \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --test_zip_feat_path temp \
    --ckpt_file save/finetune_mlm_itm_short_seq/model_step_19000.bin \
    --test_output_path save/finetune_mlm_itm_short_seq/

python -u -m torch.distributed.launch --nproc_per_node=2 vlbert_finetune_inference.py \
    --max_frames 16 --bert_seq_length 384 \
    --test_annotation /opt/ml/input/data/annotations/test.json \
    --test_zip_frames /opt/ml/input/data/zip_frames/test/ \
    --test_zip_feat_path temp \
    --ckpt_file save/finetune_mlm_itm_long_seq/model_step_19000.bin \
    --test_output_path save/finetune_mlm_itm_long_seq/
    
python merge.py
    