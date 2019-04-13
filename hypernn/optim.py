""" Contains RSGD optmizer"""
import torch
from hypernn.ops import mobius as m


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

        for group in self.param_groups:
            # no momentum
            for p in group['params']:
                if p.grad is None:
                    continue  # first step
                # p can be the various biases or the hyperbolic word
                # embeddings hence can have shape (hidden_dim,)
                # or (num_embeddings, emb_size)
                d_p = p.grad.data  # detach from comp graph
                # riemannian_grad_rescale_factor() expects a 2D
                # tensor
                unsqueezed = False
                if len(p.shape) < 2:
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
                d_p = d_p.mul(squeezed_rescale_factor)
                p.data.add_(-group['-lr'], d_p)
        return loss
