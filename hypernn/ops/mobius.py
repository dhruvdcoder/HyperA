""" Operations like mobius addition, mobius scalar mul, etc"""
import torch

##### Constants ######
ball_boundary = 1e-5
perterb = 1e-15
default_c = torch.Tensor([1.])


def dot(x, y):
    """dim(x)=batch, emb"""
    return torch.sum(x * y, dim=1, keepdim=True)


def norm(x):
    return torch.norm(x, dim=1, keepdim=True)


def norm_sq(x):
    return norm(x)**2


def atanh(x):
    """ dim(x)=any. Applies atanh to each entry"""
    x_const = torch.clamp(x, -1. + ball_boundary, 1. - ball_boundary)
    return 0.5 * torch.log((1 + x_const) / (1 - x_const))


def project_in_ball(x, c=default_c):
    """dim(x) = batch, emb"""
    # https://discuss.pytorch.org/t/how-to-use-condition-flow/644/4
    normx = norm(x)
    radius = (1. - ball_boundary) / torch.sqrt(c)
    project = x / normx * radius
    r = torch.where(normx >= radius, project, x)
    return r


def add(a, b, c=default_c):
    """Mobius a+b. dim(a)=dim(b)=batch,emb"""
    b = b + perterb
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
    sqrt_c = torch.sqrt(c)
    diff = add(-a, b, c) + perterb
    atanh_arg = sqrt_c * norm(diff)
    dist = 2. / sqrt_c * atanh(atanh_arg)
    return dist**2


def scalar_mul(r, a, c=default_c):
    """dim(r) =(1,), dim(a)=batch, emb"""
    a = a + perterb
    norm_a = norm(a)
    sqrt_c = torch.sqrt(c)
    numerator = torch.tanh(r * atanh(sqrt_c * norm_a))
    res = numerator / (sqrt_c * norm_a) * a
    return project_in_ball(res)


def conformal_factor(x, c=default_c):
    """dim(x) = batch, emb"""
    return 2. / (1. - c * dot(x, x))


def exp_map(x, v, c):
    """ x is the point on the manifold (orgin for the tangent space)
    v is a vector in the tangent space around x
    dim(x) = dim(v) = batch, emb
    """
    v = v + perterb
    norm_v = norm(v)
    sqrt_c = torch.sqrt(c)
    displacement_vector = torch.tanh(
        sqrt_c * conformal_factor(x, c) * norm_v / 2.) / (sqrt_c * norm_v) * v
    return add(x, displacement_vector)


def log_map(x, y, c):
    diff = add(-x, y, c) + perterb
    diff_n = norm(diff)
    sqrt_c = torch.sqrt(c)
    res = ((2. / (sqrt_c * conformal_factor(x, c))) * atanh(sqrt_c * diff_n) /
           diff_n) * diff
    return res


def exp_map_0(v, c):
    """special case when x=0"""
    v = v + perterb
    norm_v = norm(v)
    sqrt_c = torch.sqrt(c)
    res = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v) * v
    return project_in_ball(res, c)


def log_map_0(y, c):
    y = y + perterb
    y_n = norm(y)
    sqrt_c = torch.sqrt(c)
    return (1. / (sqrt_c * y_n)) * atanh(sqrt_c * y_n) * y


def matmul(M, x, c):
    """ 
    out = x M
    dim(x) = batch, emb
    dim(M) = emb, output
    dim(out) = batch, output
    """
    x = x + perterb
    prod = torch.matmul(x, M) + perterb
    prod_n = norm(prod)
    x_n = norm(x)
    sqrt_c = torch.sqrt(c)
    res = 1. / sqrt_c * (
        torch.tanh(prod_n / x_n * atanh(sqrt_c * x_n)) * prod) / prod_n
    return project_in_ball(res, c)
