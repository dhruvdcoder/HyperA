""" Python 3.7.2

Storing paths and configurations
"""

from pathlib import Path
import argparse
import logging
import sys
from datetime import datetime

root_dir = Path(__file__).parent.parent


def int_or_None(inp):
    if inp is None:
        return inp
    else:
        return int(inp)


model_zoo = [
    'hconcatrnn', 'hdeepavg', 'haddrnn', 'hconcatgru', 'addrnnattn',
    'hdeepavgattn'
]
rnns = ['RNN', 'GRU']


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
        help='file name of dev dataset',
        default='multinli_1.0_dev_matched.jsonl',
        type=Path)
    parser.add_argument(
        '--test_set',
        help='file name of test dataset',
        default='multinli_1.0_dev_mismatched.jsonl',
        type=Path)
    parser.add_argument(
        '--max_seq_len',
        default=None,
        help=
        'Max sentence length. If you want to truncate sentences. Default: None',
        type=int_or_None)
    parser.add_argument(
        '--vector_cache',
        type=Path,
        help='filename for saving word embeddings cache')
    parser.add_argument('--use_pretrained', default=False, action='store_true')
    parser.add_argument(
        '--emb_size',
        type=int,
        help='Has to be passed if --use_pretrained is not used')
    parser.add_argument(
        '--emb_init_avg_norm',
        default=0.001,
        type=float,
        help=('Average norm for uniformly initialized word embs.'
              'Required if --use_pretrained is False'))
    parser.add_argument(
        '--freeze_emb', action='store_true', help='Freeze embedding')
    parser.add_argument(
        '--resume_snapshot',
        type=Path,
        help='File where model snapshot is saved if resuming training'
        ' or doing test.')
    parser.add_argument(
        '--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', default=64, type=int)
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
    parser.add_argument(
        '--dtype',
        default='double',
        choices=['double', 'float'],
        help='double or float')
    parser.add_argument(
        '--mainlogdir',
        required=False,
        type=Path,
        default=(root_dir / 'logs'),
        help='Top level log dir. Logs for individual experiments are'
        ' subdirectories in this dir. (default: root_dir/logs)')
    parser.add_argument(
        '--experiment',
        default='default_experiment',
        help='name of the experiment. Use to create the logdir as'
        ' mainlogdir/experiment/')
    parser.add_argument(
        '--experiment_dir',
        type=Path,
        help='if this is passed, --mainlogdir and --experiment '
        'are ignored')
    parser.add_argument(
        '--tb_debug',
        action='store_true',
        default=True,
        help='Debug using tensorboard')
    parser.add_argument(
        '--model',
        default='hconcatgru',
        choices=model_zoo,
        help='Pick the model to train')
    parser.add_argument(
        '--rnn',
        default='RNN',
        choices=rnns,
        help='Only used when using RNN based sentence encoder')
    parser.add_argument(
        '--combine_op',
        default='add',
        choices=['add', 'concat'],
        help=
        'Method used to combine the reps of premise and hypo when using model "haddrnn".'
    )
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument(
        '--hyp_bias_lr',
        type=float,
        default=0.01,
        help=
        'Learning rate for the bias parameters in the hyperbolic space (used by RSGD)'
    )
    parser.add_argument(
        '--hyp_emb_lr',
        type=float,
        default=0.1,
        help=
        'Learning rate for the embedding parameters in the hyperbolic space (used by RSGD)'
    )
    parser.add_argument(
        '--euc_lr',
        type=float,
        default=0.001,
        help=
        'Learning rate for layers parameterized by euclidean params (used by Adam)'
    )
    parser.add_argument(
        '--print_every',
        type=int,
        default=5,
        help='Print training summary every')
    parser.add_argument(
        '--val_every',
        type=int,
        default=500,
        help='Run eval loop on dev data every val_every iterations')
    parser.add_argument(
        '--debug_grad',
        action='store_true',
        help='Pass this if debugging occurances of NaNs and Infs in gradients')
    args = parser.parse_args()
    return args


cmd_args = get_args()


# setup experiments logdir
def get_exp_dir(main_log_dir, experiment_name, timestamp):
    exp_dir = main_log_dir / '_'.join(
        [experiment_name,
         timestamp.strftime('%Y-%m-%dT%H-%M-%S')])
    return exp_dir


experiment_dir = cmd_args.experiment_dir or get_exp_dir(
    cmd_args.mainlogdir, cmd_args.experiment, datetime.now())

experiment_dir.mkdir(parents=True, exist_ok=True)

params_file = experiment_dir / 'params'
# setup root logger
file_log_handler = logging.FileHandler(experiment_dir / "run.log")
console_log_handler = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    #stream=sys.stdout,
    handlers=[file_log_handler, console_log_handler],
    level=cmd_args.logging_level,
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")
logger = logging.getLogger()

logger.info("Saving a copy of console log to {}".format(
    experiment_dir / "run.log"))

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
    save_dir = root_dir / '.saved_models'
    if cmd_args.experiment is not None:
        save_dir = save_dir / cmd_args.experiment
    elif experiment_dir.stem != '':
        save_dir = save_dir / experiment_dir.stem
    else:
        save_dir = save_dir / '.default'
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

if cmd_args.dtype == 'double':
    dtype = torch.float64
else:
    dtype = torch.float32
