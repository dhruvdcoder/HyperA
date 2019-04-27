from hypernn.models import ConcatRNN
from data.loader import prepare_multiNLI
from hypernn.training import train_main, default_params
import hypernn.embeddings as embs
import hypernn.config as config
import hypernn.models as hmodels
import torch
import argparse
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--hyp_bias_lr', type=float, default=0.01)
    parser.add_argument('--hyp_emb_lr', type=float, default=0.1)
    parser.add_argument('--euc_lr', type=float, default=0.001)
    parser.add_argument('--print_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--val_every', type=int, default=500)
    parser.add_argument('--save_dir', type=Path, default=config.save_dir)
    args, _ = parser.parse_known_args()
    return args


#train(model,
#      random_data_generator(batch_size, seq_len, num_classes, num_batches),
#      params)

if __name__ == '__main__':
    args = parse_args()
    optim_params = {
        'bias_lr': args.hyp_emb_lr,
        'emb_lr': args.hyp_emb_lr,
        'lr': args.euc_lr
    }
    test = (config.cmd_args.mode == 'test')
    data_itrs, inputs = prepare_multiNLI(
        return_test_set=test,
        embs_file=config.default_poincare_glove,
        max_seq_len=config.cmd_args.max_seq_len,
        batch_size=128,
        device=config.device,
        use_pretrained=config.cmd_args.use_pretrained)
    logger.info('Loaded data and embs')
    if config.cmd_args.resume_snapshot is not None:
        logger.info('Resuming training using model from {}'.format(
            config.cmd_args.resume_snapshot))
        model = torch.load(
            config.cmd_args.resume_snapshot,
            map_location=config.device).to(config.dtype)
    else:
        logger.info('Creating new model for training')
        logger.info('hidden_dim={}, freeze_emb={}'.format(
            args.hidden_dim, config.cmd_args.freeze_emb))
        model = ConcatRNN(
            inputs.vocab,
            args.hidden_dim,
            3,
            hmodels.default_c,
            freeze_emb=config.cmd_args.freeze_emb).to(config.dtype)
        model.to(config.device)
    train_main(
        model,
        data_itrs,
        config.cmd_args.epochs,
        optim_params,
        print_every=args.print_every,
        save_every=args.save_every,
        val_every=args.val_every,
        save_dir=args.save_dir)
