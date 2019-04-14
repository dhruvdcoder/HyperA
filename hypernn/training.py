""" Contains training loops"""
import torch
import torch.nn as nn
from hypernn import config
from hypernn.optim import RSGD
import logging
import numpy as np
import time
logger = logging.getLogger(__file__)


def default_params():
    return {'emb_lr': 0.1, 'bias_lr': 0.01, 'lr': 0.001}


#def get_model(snap_shot=config.cmd_args.model_snapshot):
#
#if args.resume_snapshot:
#    model = torch.load(args.resume_snapshot, map_location=device)
#else:
#    model = SNLIClassifier(config)
#    if args.word_vectors:
#        model.embed.weight.data.copy_(inputs.vocab.vectors)
#        model.to(device)
#
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
            logits = model.forward((batch.premise, batch.hypothesis))
            loss = loss_op(logits, batch.label)
            answer = np.argmax(logits.detach().numpy(), 1)  # shape = (batch,)
            n_correct += np.sum(answer == batch.label.detach().numpy())
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
                        loss, train_acc))
            # checkpoint model periodically
            if iterations % save_every == 0:
                snapshot_path = save_dir / 'snapshot_acc_{:.4f}_loss_{:.6f}_iter_{}_model_{}.pt'.format(
                    train_acc, loss, iterations, model.__class__.__name__)
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
                        answer = np.argmax(logits.detach().numpy(),
                                           1)  # shape = (batch,)
                        n_dev_correct += np.sum(
                            answer == dev_batch.label.detach().numpy())
                        n_dev_total += dev_batch.batch_size
                        dev_loss = loss_op(logits, dev_batch.label)
                    dev_acc = 100. * n_dev_correct / n_dev_total
                logger.info(config.dev_log_header)
                logger.info(
                    config.dev_log_template.format(
                        time.time() - start, epoch, num_epochs, iterations,
                        1 + batch_idx, len(train_itr),
                        100. * (1 + batch_idx) / len(train_itr), loss,
                        dev_loss, train_acc, dev_acc))

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
