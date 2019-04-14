""" Python 3.7.2

Storing paths and configurations
"""

from pathlib import Path
import argparse
import logging
import sys


def get_args():
    parser = argparse.ArgumentParser(
        description='To level config for the HyperA project')
    parser.add_argument(
        '-r',
        '--root_dir',
        type=Path,
        default=Path(__file__).parent.parent,
        help='Path to root directory for the project.')
    parser.add_argument(
        '-s', '--save_to_latex', action='store_true', default=False)
    parser.add_argument(
        '-l',
        '--logging_level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        type=lambda s: logging.getLevelName(s))
    parser.add_argument(
        '--dataset',
        choices=['snli', 'multinli'],
        default='multinli',
        help='One of snli or multinli')
    parser.add_argument(
        '--train_set',
        help='file name of train dataset',
        default='multinli_1.0_train.jsonl',
        type=Path)
    parser.add_argument(
        '--val_set',
        help='file name of train dataset',
        default='multinli_1.0_dev_matched.jsonl',
        type=Path)
    parser.add_argument(
        '--test_set',
        help='file name of test dataset',
        default='multinli_1.0_dev_mismatched.jsonl',
        type=Path)
    parser.add_argument(
        '--max_seq_len', default=100, help='Max sentence length', type=int)
    parser.add_argument(
        '--vector_cache',
        type=Path,
        help='filename for saving word embeddings cache')
    parser.add_argument(
        '--resume_snapshot',
        type=Path,
        help='File where model snapshot is saved if resuming training'
        ' or doing test.')
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument(
        '--save_dir',
        type=Path,
        help='Dir where to save model snaps models are saved')
    parser.add_argument(
        '--save_every',
        default=1000,
        type=int,
        help='Checkpoint saving after how many iters')
    parser.add_argument('--mode', choices=['test', 'train'], default='train')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument(
        '--gpu', default=0, type=int, help="Used only if device is gpu")
    args, _ = parser.parse_known_args()
    return args


cmd_args = get_args()

root_dir = Path(__file__).parent.parent

logging.basicConfig(
    stream=sys.stdout,
    level=cmd_args.logging_level,
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")

data_dir = root_dir / 'data'

dataset_dir = data_dir / cmd_args.dataset

code_dir = root_dir / 'hypernn'

test_dir = root_dir / 'test'

report_dir = root_dir / 'report'

emb_dir = root_dir / 'embeddings'

default_poincare_glove = emb_dir / 'poincare' / 'poincare_glove_100D_cosh-dist-sq_init_trick_word2vec.txt'

if cmd_args.vector_cache is not None:
    emb_cache_file = str(cmd_args.vector_cache)
else:
    (emb_dir / '.vector_cache').mkdir(parents=True, exist_ok=True)
    emb_cache_file = emb_dir / '.vector_cache' / 'emb_cache.pt'

#if cmd_args.model_snapshot is not None:
#    model_snapshot = cmd_args.model_snapshot
#else:
#    _temp = root_dir / '.model_snaps'
#    _temp.mkdir(parents=True, exists_ok=True)
#    model_snapshot = root_dir / '.model_snaps' / 'default_snap.pt'

if cmd_args.save_dir is not None:
    save_dir = cmd_args.save_dir
else:
    save_dir = root_dir / '.saved_models' / '.default'
save_dir.mkdir(parents=True, exist_ok=True)

train_log_header = '  Time Epoch/Total Iteration Loss Accuracy'
dev_log_header = '  Time Epoch/Total Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join(
    '{:>6.0f},{:>5.0f}/{},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'
    .split(','))
train_log_template = ' '.join(
    '{:>6.0f},{:>5.0f}/{},{:>9.0f},{:>8.6f},{:12.4f}'.split(','))

import torch
if cmd_args.device == 'cpu':
    device = torch.device('cpu')
else:
    device = torch.device('cuda:{}'.format(cmd_args.gpu))
