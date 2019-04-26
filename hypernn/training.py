""" Contains training loops"""
import torch
import torch.nn as nn
from hypernn import config
from hypernn.optim import RSGD
import logging
import numpy as np
import time
import logger as tb_logger
import argparse
from pathlib import Path
from data.loader import prepare_multiNLI
from hypernn.models import ConcatRNN, HyperDeepAvgNet, AddRNN, default_c
logger = logging.getLogger(__file__)


def default_params():
    return {'emb_lr': 0.1, 'bias_lr': 0.01, 'lr': 0.001}


model_zoo = {
    'hconcatrnn': ConcatRNN,
    'hdeepavg': HyperDeepAvgNet,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        default='hconcatrnn',
        choices=list(model_zoo.keys()),
        help=' or '.join(list(model_zoo.keys())))
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


def train(model, data_gen, params):
    loss_op = nn.CrossEntropyLoss()
    hyp_param_gropus = model.get_hyperbolic_params(
        emb_lr=params['emb_lr'], bias_lr=params['bias_lr'])
    euc_param_groups = model.get_euclidean_params(lr=params['lr'])
    eu_optim = torch.optim.Adam(euc_param_groups)
    hyp_optim = RSGD(hyp_param_gropus, model.c)
    for inp, label in data_gen():
        logits = model.forward(inp)
        eu_optim.zero_grad()
        hyp_optim.zero_grad()
        loss = loss_op(logits, label)
        logger.info("loss: {}".format(loss))
        loss.backward()
        eu_optim.step()
        hyp_optim.step()


def train_main(model,
               data,
               num_epochs,
               optim_params,
               print_every=1000,
               save_every=10000,
               val_every=10,
               save_dir=config.save_dir):
    train_itr, dev_itr, test_itr = data
    loss_op = nn.CrossEntropyLoss()
    hyp_param_gropus = model.get_hyperbolic_params(
        emb_lr=optim_params['emb_lr'], bias_lr=optim_params['bias_lr'])
    euc_param_groups = model.get_euclidean_params(lr=optim_params['lr'])
    eu_optim = torch.optim.Adam(euc_param_groups)
    hyp_optim = RSGD(hyp_param_gropus, model.c)
    logger.info('Setting up optimizers {} and {} with params {}'.format(
        eu_optim.__class__.__name__, hyp_optim.__class__.__name__,
        optim_params))
    iterations = 0
    start = time.time()
    best_dev_acc = -1
    dev_acc = -2
    for epoch in range(num_epochs):
        train_itr.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in enumerate(train_itr):

            model.train()
            eu_optim.zero_grad()
            hyp_optim.zero_grad()

            iterations += 1
            # plot model to tensorboard
            # This does not work yet because onnx
            # does not support unbind and frobenius_norm
            #tb_logger.tb_logger.add_graph(model,
            #                              [batch.premise, batch.hypothesis])
            logits = model.forward((batch.premise, batch.hypothesis))
            loss = loss_op(logits, batch.label)
            #answer = np.argmax(logits.detach().numpy(), 1)  # shape = (batch,)
            with torch.no_grad():
                answer = torch.max(logits, 1)[1]
                #n_correct += np.sum(answer == batch.label.detach().numpy())
                n_correct += (answer == batch.label).sum().item()
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total
            loss.backward()
            eu_optim.step()
            hyp_optim.step()

            # pring train log
            if iterations % print_every == 0:
                logger.info(config.train_log_header)
                logger.info(
                    config.train_log_template.format(
                        time.time() - start, epoch, num_epochs, iterations,
                        loss.item(), train_acc))
                # print the tensorboard
                tb_logger.tb_logger.add_scalar('loss', loss.item(), iterations)
                tb_logger.tb_logger.add_scalar('train_acc', train_acc,
                                               iterations)
            # checkpoint model periodically
            if iterations % save_every == 0:
                snapshot_path = save_dir / 'snapshot_acc_{:.4f}_loss_{:.6f}_iter_{}_model_{}.pt'.format(
                    train_acc, loss.item(), iterations,
                    model.__class__.__name__)
                logger.info('Saving periodic snap to {}'.format(snapshot_path))
                torch.save(model, snapshot_path)
                # remove old snaps
                for f in save_dir.glob('snapshot' + '*'):
                    if f != snapshot_path:
                        f.unlink()

            #eval
            if iterations % val_every == 0:
                # switch model to evaluation mode
                model.eval()
                dev_itr.init_epoch()
                n_dev_correct, n_dev_total = 0, 0
                with torch.no_grad():
                    for dev_batch_idx, dev_batch in enumerate(dev_itr):
                        logits = model((dev_batch.premise,
                                        dev_batch.hypothesis))
                        #answer = np.argmax(logits.detach().numpy(),
                        #                  1)  # shape = (batch,)
                        answer = torch.max(logits, 1)[1]
                        #n_dev_correct += np.sum(
                        #    answer == dev_batch.label.detach().numpy())
                        n_dev_correct += (
                            answer == dev_batch.label).sum().item()
                        n_dev_total += dev_batch.batch_size
                        dev_loss = loss_op(logits, dev_batch.label)
                    dev_acc = 100. * n_dev_correct / n_dev_total
                logger.info(config.dev_log_header)
                logger.info(
                    config.dev_log_template.format(
                        time.time() - start, epoch, num_epochs, iterations,
                        1 + batch_idx, len(train_itr),
                        100. * (1 + batch_idx) / len(train_itr), loss,
                        dev_loss.item(), train_acc, dev_acc))
                tb_logger.tb_logger.add_scalar('dev_acc', dev_acc, iterations)
                # update best valiation set accuracy
                if dev_acc > best_dev_acc:

                    # found a model with better validation set accuracy

                    best_dev_acc = dev_acc
                    snapshot_path = save_dir / 'best_snapshot_devacc_{}_devloss_{}__iter_{}_model_{}.pt'.format(
                        dev_acc, dev_loss, iterations,
                        model.__class__.__name__)
                    logger.info(
                        'Saving best model in {}'.format(snapshot_path))
                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)
                    for f in save_dir.glob('best_snapshot' + '*'):
                        if f != snapshot_path:
                            f.unlink()


def get_model(args, inputs):
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
        model = model_zoo[args.model](
            inputs.vocab,
            args.hidden_dim,
            3,
            default_c,
            freeze_emb=config.cmd_args.freeze_emb,
            emb_size=config.cmd_args.emb_size,
            init_avg_norm=config.cmd_args.emb_init_avg_norm).to(config.dtype)
        model.to(config.device)
        logger.info("Using model: {}".format(model.__class__.__name__))
    return model


if __name__ == '__main__':
    args = parse_args()
    optim_params = {
        'bias_lr': args.hyp_bias_lr,
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
        model = get_model(args, inputs)
    train_main(
        model,
        data_itrs,
        config.cmd_args.epochs,
        optim_params,
        print_every=args.print_every,
        save_every=args.save_every,
        val_every=args.val_every,
        save_dir=args.save_dir)
