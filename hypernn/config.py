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
    args, _ = parser.parse_known_args()
    return args


cmd_args = get_args()

root_dir = Path(__file__).parent.parent

#logging_level = cmd_args.logging_level or 'DEBUG'
logging_level = 'DEBUG'

logging.basicConfig(
    stream=sys.stdout,
    level=logging_level,
    format="[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s")

data_dir = root_dir.parent / 'data'

code_dir = root_dir / 'hypernn'

test_dir = root_dir / 'test'

report_dir = root_dir / 'report'

emb_dir = root_dir / 'embeddings'

default_poincare_glove = emb_dir / 'poincare' / 'poincare_glove_100D_dist-sq_vocab50k_word2vec.txt'
