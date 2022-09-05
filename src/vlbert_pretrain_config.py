import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline for Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    
    parser.add_argument('--use_mfm', default=False, type=bool)
    
    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='/opt/ml/input/data/annotations/labeled.json')
    parser.add_argument('--train_annotation_unlabeled', type=str, default='/opt/ml/input/data/annotations/unlabeled.json')
    
    parser.add_argument('--train_zip_feat_path', type=str, default='save/zip_feats/labeled_swin_tiny.zip')
    parser.add_argument('--train_zip_feat_path_unlabeled', type=str, default='save/zip_feats/unlabeled_swin_tiny.zip')
    
    parser.add_argument('--batch_size', default=110, type=int, help="use for training duration per worker")
    
    parser.add_argument('--train_prefetch_factor', default=2, type=int, help="use for training duration per worker")
    parser.add_argument('--train_num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--model_name', type=str, default='0718_vlbert_v113_freeze_swin')
    parser.add_argument('--ckpt_file', type=str, default='save/model.bin')
    parser.add_argument('--pretrain_ckpt_file', type=str, default=None)
    parser.add_argument('--best_score', default=0.68, type=float, help='save checkpoint if mean_f1 > best_score')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=20, help='How many epochs')
    parser.add_argument('--print_steps', type=int, default=200, help="Number of steps to log training metrics.")
    parser.add_argument('--minimum_val_steps', type=int, default=10000)
    parser.add_argument('--val_steps', type=int, default=10000)
    parser.add_argument('--save_steps', type=int, default=110000)
    parser.add_argument('--warmup_ratio', default=0.06, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    
    parser.add_argument('--default_learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
    parser.add_argument('--swin_learning_rate', type=float, default=3e-5)
    
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== Swin ===================================
    parser.add_argument('--swin_pretrained_path', type=str, default='/opt/ml/wxcode/opensource_models/swinv2_tiny_patch4_window8_256.pth')

    # ========================== Title BERT =============================
    parser.add_argument('--bert_dir', type=str, default='/opt/ml/wxcode/opensource_models/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    parser.add_argument('--bert_seq_length', type=int, default=256)
    
    parser.add_argument('--bert_layerwise_learning_rate_decay', type=float, default=0.975)
    parser.add_argument('--bert_freezing_layers', type=int, default=0)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=16)

    return parser.parse_args()
