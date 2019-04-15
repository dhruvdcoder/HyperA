""" Contains RSGD optmizer"""
import torch
from hypernn.ops import mobius as m
from hypernn import config as config
import logging
logger = logging.getLogger(__name__)


def riemannian_grad_rescale_factor(u, c):
    """ Since, the poincare ball model and Rn are conformal, the riemannian grad 
    is just a rescaled version of the Euclidian grad. (1/lamnbda(u))^2

    See eq. 36
    in paper "Hyperbolic entailment cones for learning hierarchical embeddings"
    for derivation. The output of this function has to be multiplied by the
    euclidian grad to obtain the grad in poincare ball model.

    Arguments:

        u : Point in hyperbolic space with shape=batch, hidden_dim

        c : c

    Returns:
        factor: Shape=batch,1
    """
    return ((1. - c * m.dot(u, u))**2) / 4.0


class RSGD(torch.optim.Optimizer):
    def __init__(self, params, c, lr=0.001):
        defaults = dict(lr=lr)
        super(RSGD, self).__init__(params, defaults)
        self.c = c

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        group_no = 0
        for group in self.param_groups:
            logger.debug("Group num={}".format(group_no))
            # no momentum
            param_num = 0
            for p in group['params']:
                logger.debug("param_num={}".format(param_num))
                logger.debug("shape={}".format(p.shape))
                if p.grad is None:
                    continue  # first step
                # p can be the various biases or the hyperbolic word
                # embeddings hence can have shape (hidden_dim,)
                # or (num_embeddings, emb_size)
                d_p = p.grad.data  # detach from comp graph
                logger.debug("E grad={}".format(d_p))
                # riemannian_grad_rescale_factor() expects a 2D
                # tensor
                unsqueezed = False
                if len(p.shape) < 2:  # for biases
                    if len(p.shape) == 1:
                        reshaped_p = p.detach().unsqueeze(0)
                        unsqueezed = True
                    else:
                        raise ValueError(
                            "Invalid shape of hyperbolic parameter {}".format(
                                p))
                else:
                    reshaped_p = p.detach()
                rescale_factor = riemannian_grad_rescale_factor(
                    reshaped_p, self.c)
                if unsqueezed:
                    squeezed_rescale_factor = rescale_factor.squeeze(0)
                else:
                    squeezed_rescale_factor = rescale_factor

                # apply riemannian gradient
                # Be careful while changing operators here
                # because embeddings might be sparse matrices
                # hence only those operators are allowed which
                # are supported by sparse tensors

                #### following does not work because
                #### Sparse*Dense does not support
                #### broadcasting on dense
                #### https://github.com/pytorch/pytorch/issues/3158
                #### use the following line if embeddings are not sparse
                #### If embeddings are sparse, use the code block below
                #### this line
                d_p = d_p.mul(squeezed_rescale_factor)
                logger.debug("H grad={}".format(d_p))
                #if len(d_p.shape) == 2:
                #    d_p = d_p.mul(
                #       squeezed_rescale_factor.expand(-1, d_p.size(1)))
                #else:
                #   d_p = d_p.mul(squeezed_rescale_factor.expand(d_p.size(0)))

                #p.data.add_(-group['lr'],
                #           d_p)  # is this correct or should we do
                # mobius add?
                p.data = m.exp_map(p.data, -group['lr'] * d_p, m.default_c)
                param_num += 1
            group_no += 1
        return loss
