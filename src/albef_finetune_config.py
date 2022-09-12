import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_parallel', default=False, type=bool)

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--test_annotation', type=str, default='data/annotations/test_b.json')
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--test_zip_feats', type=str, default='data/zip_feats/test_b.zip')
    parser.add_argument('--test_output_path', type=str, default='')
    
    parser.add_argument('--use_val', default=False, type=bool)
    parser.add_argument('--n_splits', default=10, type=int)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help="use for training duration per worker")
    parser.add_argument('--val_batch_size', default=256, type=int, help="use for validation duration per worker")
    parser.add_argument('--test_batch_size', default=256, type=int, help="use for testing duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=4, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='')
    parser.add_argument('--ckpt_file', type=str,default='')
    parser.add_argument('--pretrain_ckpt_file', type=str, default='')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=4, help='How many epochs')
    parser.add_argument('--print_steps', type=int, default=200, help="Number of steps to log training metrics.")
    parser.add_argument('--val_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=9000)
    
    parser.add_argument('--warmup_ratio', default=0.06, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument('--use_fgm', type=bool, default=False)
    parser.add_argument('--use_ema', type=bool, default=False)

    # ========================== BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    
    parser.add_argument('--bert_seq_length_concat', type=int, default=384)
    parser.add_argument('--bert_learning_rate', type=float, default=2e-5)
    parser.add_argument('--bert_layerwise_learning_rate_decay', type=float, default=0.975)
    parser.add_argument('--bert_freezing_layers', type=int, default=0)

    # ========================== Video =============================
    parser.add_argument('--use_vlad', type=bool, default=True)
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--vlad_cluster_size', type=int, default=64)
    parser.add_argument('--vlad_groups', type=int, default=8)
    parser.add_argument('--vlad_hidden_size', type=int, default=1024, help='nextvlad output size using dense')
    parser.add_argument('--se_ratio', type=int, default=8, help='reduction factor in se context gating')
    parser.add_argument('--use_vision_bert_emb', type=bool, default=False)

    # ========================== Fusion Layer =============================
    parser.add_argument('--fc_size', type=int, default=768, help="linear size before final linear")
    parser.add_argument('--classifier_mlp_sizes', nargs='+', type=int, default=[768])

    # ========================== ALBEF =============================
    parser.add_argument('--vision_layer_num', type=int, default=1)

    return parser.parse_args()
