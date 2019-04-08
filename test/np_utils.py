import numpy as np
import logging
import torch
logger = logging.getLogger(__name__)

##### Constants ######
ball_boundary = 1e-5
perterb = 1e-15
default_c = torch.Tensor([1.])


def random_vec(size, low=-0.01, high=0.01):
    return np.random.uniform(low=low, high=high, size=size)


def dot(x, y):
    """dim(x)=batch, emb"""
    return np.sum(x * y, axis=1)[:, None]


def norm(x):
    return np.linalg.norm(x, axis=1)[:, None]


def norm_sq(x):
    return norm(x)**2


def atanh(x):
    """ dim(x)=any. Applies atanh to each entry"""
    x_const = np.clip(x, -1. + ball_boundary, 1. - ball_boundary)
    return np.arctanh(x_const)


def project_in_ball(x, c=default_c):
    """dim(x) = batch, emb"""
    # https://discuss.pytorch.org/t/how-to-use-condition-flow/644/4
    normx = norm(x)
    radius = (1. - ball_boundary) / np.sqrt(c)
    project = x / normx * radius
    r = np.where(normx >= radius, project, x)
    return r


def add(a, b, c):
    b += perterb
    norm_sq_a = c * norm_sq(a)
    norm_sq_b = c * norm_sq(b)
    inner_ab = c * dot(a, b)
    c1 = 1. + 2. * inner_ab + norm_sq_b
    c2 = 1. - norm_sq_a
    d = 1. + 2. * inner_ab + norm_sq_b * norm_sq_a
    res = c1 / d * a + c2 / d * b
    return project_in_ball(res, c=c)


def squared_distance(a, b, c=default_c):
    """dim(a)=dim(b)=batch, emb
    dim(output)=batch,1"""
    sqrt_c = np.sqrt(c)
    diff = add(-a, b, c) + perterb
    atanh_arg = sqrt_c * norm(diff)
    dist = 2. / sqrt_c * atanh(atanh_arg)
    return dist**2
