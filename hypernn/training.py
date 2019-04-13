""" Contains training loops"""
import torch
import torch.nn as nn
from hypernn.optim import RSGD
import logging
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
