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
from hypernn.models import model_zoo, default_c
logger = logging.getLogger(__file__)


def default_params():
    return {'emb_lr': 0.1, 'bias_lr': 0.01, 'lr': 0.001}

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


def test_main(model,
               inputs,
               answers,
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
    #output=[]
    start = time.time()
    best_dev_acc = -1
    dev_acc = -2
    output_file = config.experiment_dir / "predictions_outfile.txt"
    with open(output_file, 'a+') as f:
        f.write('Premise|hypothesis|True|Pred\n')
    test_itr.init_epoch()
    n_dev_correct, n_dev_total = 0, 0
    pad_idx = inputs.vocab.stoi['<pad>']
    eos_idx = inputs.vocab.stoi['<eos>']
    sos_idx = inputs.vocab.stoi['<sos>']
    for batch_idx, dev_batch in enumerate(test_itr):

        #eval
        output=[]
        predictions=[]
        hypothesis=[]
        premise=[]

        model.eval()
        dev_itr.init_epoch()
        logits = model((dev_batch.premise,
                        dev_batch.hypothesis))
        #answer = np.argmax(logits.detach().numpy(),
        #                  1)  # shape = (batch,)
        answer = torch.max(logits, 1)[1]
        for pred in answer.tolist():
            predictions.append(answers.vocab.itos[pred])
        for out in dev_batch.label.tolist():
            output.append(answers.vocab.itos[out])
        for prem in dev_batch.premise.tolist():
            sent = []
            for word in prem:
                if inputs.vocab.itos[word] == '<pad>':
                    break
                sent.append(inputs.vocab.itos[word])
            premise.append(' '.join(sent))

        for hyp in dev_batch.hypothesis.tolist():
            sent = []
            for word in hyp:
                if inputs.vocab.itos[word] == '<pad>':
                    break
                sent.append(inputs.vocab.itos[word])
            hypothesis.append(' '.join(sent))

        with open(output_file, 'a+') as f:
            for item in range(len(dev_batch)):
                f.write('|'.join([premise[item], hypothesis[item], output[item], predictions[item]+'\n']))

        n_dev_correct += (
            answer == dev_batch.label).sum().item()
        n_dev_total += dev_batch.batch_size

    dev_acc = 100. * n_dev_correct / n_dev_total


def get_model(args, inputs):
    if args.resume_snapshot is not None:
        logger.info('Resuming training using model from {}'.format(
            args.resume_snapshot))
        model = torch.load(
            args.resume_snapshot, map_location=config.device).to(config.dtype)
    else:
        logger.info('Creating new model for training')
        logger.info('hidden_dim={}, freeze_emb={}'.format(
            args.hidden_dim, args.freeze_emb))
        model = model_zoo[args.model](
            inputs.vocab,
            args.hidden_dim,
            3,
            default_c,
            rnn=args.rnn,
            freeze_emb=args.freeze_emb,
            emb_size=args.emb_size,
            init_avg_norm=args.emb_init_avg_norm).to(config.dtype)
        model.to(config.device)
        logger.info("Using model: {}".format(model.__class__.__name__))
    return model


if __name__ == '__main__':
    args = config.cmd_args
    optim_params = {
        'bias_lr': args.hyp_bias_lr,
        'emb_lr': args.hyp_emb_lr,
        'lr': args.euc_lr
    }
    test = (args.mode == 'test')
    if test == True:
        data_itrs, inputs,answers = prepare_multiNLI(
        return_test_set=test,
        embs_file=config.default_poincare_glove,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        device=config.device,
        use_pretrained=args.use_pretrained)
    else:
        data_itrs, inputs = prepare_multiNLI(
            return_test_set=test,
            embs_file=config.default_poincare_glove,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            device=config.device,
            use_pretrained=args.use_pretrained)
    logger.info('Loaded data and embs')
    if args.resume_snapshot is not None:
        logger.info('Resuming training using model from {}'.format(
            args.resume_snapshot))
        model = torch.load(
            args.resume_snapshot, map_location=config.device).to(config.dtype)
    else:
        model = get_model(args, inputs)
    test_main(
        model,
        inputs,
        answers,
        data_itrs,
        args.epochs,
        optim_params,
        print_every=args.print_every,
        save_every=args.save_every,
        val_every=args.val_every,
        save_dir=config.save_dir)
