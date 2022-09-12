import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Weixin Challenge 2022")

    parser.add_argument("--seed", type=int, default=2022, help="random seed.")
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--use_parallel', default=False, type=bool)

    parser.add_argument('--use_mlm', default=True, type=bool)
    parser.add_argument('--use_itm', default=True, type=bool)

    # ========================= Data Configs ==========================
    parser.add_argument('--train_annotation', type=str, default='data/annotations/labeled.json')
    parser.add_argument('--train_annotation_unlabeled', type=str, default='data/annotations/unlabeled.json')
    
    parser.add_argument('--train_zip_feats', type=str, default='data/zip_feats/labeled.zip')
    parser.add_argument('--train_zip_feats_unlabeled', type=str, default='data/zip_feats/unlabeled.zip')

    parser.add_argument('--batch_size', default=110, type=int, help="use for training duration per worker")
    parser.add_argument('--prefetch', default=16, type=int, help="use for training duration per worker")
    parser.add_argument('--num_workers', default=12, type=int, help="num_workers for dataloaders")

    # ======================== SavedModel Configs =========================
    parser.add_argument('--savedmodel_path', type=str, default='')

    # ========================= Learning Configs ==========================
    parser.add_argument('--max_epochs', type=int, default=30, help='How many epochs')
    parser.add_argument('--print_steps', type=int, default=200, help="Number of steps to log training metrics.")
    parser.add_argument('--save_steps', type=int, default=100000)
    
    parser.add_argument('--warmup_ratio', default=0.06, type=int, help="warm ups for parameters not in bert or vit")
    parser.add_argument('--minimum_lr', default=0., type=float, help='minimum learning rate')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    # ========================== BERT =============================
    parser.add_argument('--bert_dir', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--bert_output_size', type=int, default=768)
    parser.add_argument('--bert_cache', type=str, default='data/cache')
    
    parser.add_argument('--bert_seq_length_concat', type=int, default=384)
    parser.add_argument('--bert_learning_rate', type=float, default=5e-5)
    parser.add_argument('--bert_layerwise_learning_rate_decay', type=float, default=1.0)

    # ========================== Video =============================
    parser.add_argument('--frame_embedding_size', type=int, default=768)
    parser.add_argument('--max_frames', type=int, default=32)
    parser.add_argument('--use_vision_bert_emb', type=bool, default=True)

    # ========================== ALBEF =============================
    parser.add_argument('--vision_layer_num', type=int, default=1)

    return parser.parse_args()
