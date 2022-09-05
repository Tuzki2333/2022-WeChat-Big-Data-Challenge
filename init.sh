cd /opt/ml/wxcode/src

python extract_feature_train.py \
    --zip_frame_dir /opt/ml/input/data/zip_frames/labeled/ \
    --ann_path /opt/ml/input/data/annotations/labeled.json \
    --output_path save/zip_feats/labeled_swin_tiny.zip
    
python extract_feature_train.py \
    --max_frames 32 \
    --zip_frame_dir /opt/ml/input/data/zip_frames/unlabeled/ \
    --ann_path /opt/ml/input/data/annotations/unlabeled.json \
    --output_path save/zip_feats/unlabeled_swin_tiny.zip