""" Operations like mobius addition, mobius scalar mul, etc"""
import torch

inner = torch.dot
norm_sq = lambda x: torch.norm(x)**2


def project_in_ball(x, c=1.):
    # https://discuss.pytorch.org/t/how-to-use-condition-flow/644/4
    normx = torch.norm(x)
    if normx >= c:
        r = x / normx * c
    else:
        r = x
    return r


def add(a, b, c=1.):
    """Mobius a+b"""
    norm_sq_a = c * norm_sq(a)
    norm_sq_b = c * norm_sq(b)
    inner_ab = c * inner(a, b)
    c1 = 1. + 2. * inner_ab + norm_sq_b
    c2 = 1. - norm_sq_a
    d = 1. + 2. * inner_ab + norm_sq_b * norm_sq_a
    res = c1 / d * a + c2 / d * b
    return project_in_ball(res, c=c)
