""" Operations like mobius addition, mobius scalar mul, etc"""
import torch
from numpy import sqrt

##### Constants ######
ball_boundary = 1e-5
perterb = 1e-15
default_c = 1.


def dot(x, y, dim=-1):
    """dim(x)=batch, emb"""
    return torch.sum(x * y, dim=dim, keepdim=True)


def norm(x, dim=-1):
    return torch.norm(x, dim=dim, keepdim=True)


def norm_sq(x, dim=-1):
    return norm(x, dim=dim)**2


def atanh(x):
    """ dim(x)=any. Applies atanh to each entry"""
    x_const = torch.clamp(x, -1. + ball_boundary, 1. - ball_boundary)
    return 0.5 * torch.log((1 + x_const) / (1 - x_const))


def asinh(x):
    """ dim(x)=any. Applies asinh to each entry"""
    return torch.log(x + (x**2 + 1)**0.5)


def project_in_ball(x, c=default_c, dim=-1):
    """dim(x) = batch, *, *,emb"""
    # https://discuss.pytorch.org/t/how-to-use-condition-flow/644/4
    normx = norm(x, dim=dim)
    radius = (1. - ball_boundary) / sqrt(c)
    project = x / normx * radius
    r = torch.where(normx >= radius, project, x)
    return r


def add(a, b, c=default_c, dim=-1):
    """Mobius a+b. dim(a)=dim(b)=batch,**,emb"""
    b = b + perterb
    norm_sq_a = c * norm_sq(a, dim=dim)
    norm_sq_b = c * norm_sq(b, dim=dim)
    inner_ab = c * dot(a, b, dim=dim)
    c1 = 1. + 2. * inner_ab + norm_sq_b
    c2 = 1. - norm_sq_a
    d = 1. + 2. * inner_ab + norm_sq_b * norm_sq_a
    res = c1 / d * a + c2 / d * b
    return project_in_ball(res, c=c, dim=dim)


def squared_distance(a, b, c=default_c, dim=-1):
    """dim(a)=dim(b)=batch, **,emb
    dim(output)=batch,1"""
    sqrt_c = sqrt(c)
    diff = add(-a, b, c, dim=dim) + perterb
    atanh_arg = sqrt_c * norm(diff, dim=dim)
    dist = 2. / sqrt_c * atanh(atanh_arg)
    return dist**2


def scalar_mul(r, a, c=default_c, dim=-1):
    """dim(r) =(1,), dim(a)=batch, emb"""
    a = a + perterb
    norm_a = norm(a, dim=dim)
    sqrt_c = sqrt(c)
    numerator = torch.tanh(r * atanh(sqrt_c * norm_a))
    res = numerator / (sqrt_c * norm_a) * a
    return project_in_ball(res, dim=dim)


def conformal_factor(x, c=default_c, dim=-1):
    """dim(x) = batch, **,emb"""
    return 2. / (1. - c * dot(x, x, dim=dim))


def exp_map(x, v, c, dim=-1):
    """ x is the point on the manifold (orgin for the tangent space)
    v is a vector in the tangent space around x
    dim(x) = dim(v) = batch,**, emb
    """
    v = v + perterb
    norm_v = norm(v, dim=dim)
    sqrt_c = sqrt(c)
    displacement_vector = torch.tanh(sqrt_c * conformal_factor(x, c, dim=dim) *
                                     norm_v / 2.) / (sqrt_c * norm_v) * v
    return add(x, displacement_vector, dim=dim)


def log_map(x, y, c, dim=-1):
    diff = add(-x, y, c, dim=dim) + perterb
    diff_n = norm(diff, dim=dim)
    sqrt_c = sqrt(c)
    res = ((2. / (sqrt_c * conformal_factor(x, c, dim=dim))) * atanh(
        sqrt_c * diff_n) / diff_n) * diff
    return res


def exp_map_0(v, c, dim=-1):
    """special case when x=0"""
    v = v + perterb
    norm_v = norm(v, dim=-1)
    sqrt_c = sqrt(c)
    res = torch.tanh(sqrt_c * norm_v) / (sqrt_c * norm_v) * v
    return project_in_ball(res, c, dim=dim)


def log_map_0(y, c, dim=-1):
    y = y + perterb
    y_n = norm(y, dim=-1)
    sqrt_c = sqrt(c)
    return (1. / (sqrt_c * y_n)) * atanh(sqrt_c * y_n) * y


def matmul(M, x, c, dim=-1):
    """ 
    out = x M
    dim(x) = batch, **,emb
    dim(M) = emb, output
    dim(out) = batch, output
    """
    x = x + perterb
    prod = torch.matmul(x, M) + perterb
    prod_n = norm(prod, dim=dim)
    x_n = norm(x, dim=dim)
    sqrt_c = sqrt(c)
    res = 1. / sqrt_c * (
        torch.tanh(prod_n / x_n * atanh(sqrt_c * x_n)) * prod) / prod_n
    return project_in_ball(res, c, dim=dim)


def sum(x, c, dim=-2):
    """Not starightforward to vectorize add.
    Hence we loop
        TODO: Vectorize

    Arguments:

        x: Tensor of len(shape) >=2

        dim: Dimension to sum along. In most cases, expecting
        shape (batch, seq, hidden)
        and we will sum along seq by default

    """
    ##########
    #b = b + perterb
    ##norm_sq_a = c * norm_sq(a, dim=dim)
    #norm_sq_b = c * norm_sq(b, dim=dim)
    #inner_ab = c * dot(a, b, dim=dim)
    #c1 = 1. + 2. * inner_ab + norm_sq_b
    #c2 = 1. - norm_sq_a
    #d = 1. + 2. * inner_ab + norm_sq_b * norm_sq_a
    #res = c1 / d * a + c2 / d * b
    #return project_in_ball(res, c=c, dim=dim)
    out_shape = list(x.size())
    out_shape.pop(dim)
    acc = torch.zeros(out_shape)
    for seq_arr in torch.unbind(x, dim):
        # seq_arr will be of shape (batch, hidden)
        acc = add(acc, seq_arr, c)
    return acc


def mean(x, c, dim=-2):
    """ 
    Arguments:

        x: Tensor of len(shape) >=2

        dim: Dimension to sum along. In most cases, expecting
        shape (batch, seq, hidden)
        and we will sum along seq by default

    """
    n = x.size(dim)
    s = sum(x, c, dim=dim)

    new_hidden_dim_idx = dim + 1
    r = scalar_mul(1. / n, s, c, dim=new_hidden_dim_idx)
    return r


def logits(x, p, a, c):
    """Finds the logits to be used by softmax

    Arguments:

        x : Input tensor with shape (batch, hidden_dim)

        p : Parameter matrix of hyperbolic MLR with shape
            (num_classes, hidden_dim) (see eq. 25 in HNN paper)

        a : Parameter matrix of hyperbolic MLR with shape
            (num_classes, hidden_dim) (see eq. 25 in HNN paper)
        
        c : c
    """
    # Because our mobius operations are only
    # defined for 2-dimensions, will form the
    # logit matrix of shape (batch, num_classes)
    # column by column
    assert p.shape == a.shape
    assert p.size(1) == x.size(1)
    dot_px_as = []
    cf_pxs = []
    norm_a = []
    for col_p, col_a in zip(torch.unbind(p, dim=0), torch.unbind(a, dim=0)):
        minus_p_plus_x = add(-col_p[None, :], x, c)  # shape=batch, hidden
        cf_px = conformal_factor(minus_p_plus_x, c)  # shape=batch,1
        cf_pxs.append(cf_px.squeeze(1))
        a_norm = torch.norm(col_a)
        norm_a.append(a_norm)
        col_a = col_a[None, :]
        dot_px_a = dot(
            minus_p_plus_x,
            col_a / a_norm,
        )  # shape=batch,1
        dot_px_as.append(dot_px_a.squeeze(1))
    cfs = torch.stack(cf_pxs, dim=1)  # shape=batch, num_classes
    norm_a = torch.stack(norm_a)[None, :]  # shape=1,num_classes
    dots = torch.stack(dot_px_as, dim=1)  # shape=batch, num_classes
    sqrt_c = sqrt(c)
    logits = 2. / sqrt_c * norm_a * asinh(sqrt_c * (dots * cfs))
    return logits


def activation(x, function, c):
    """ Applies mobius version of map by using exponential 
    and logarithamic maps (see eq. 26 in HNN paper)

    Arguments:

        x : Input tensor

        function: One of the non-linearities like relu, tanh, etc

    """
    return exp_map_0(function(log_map_0(x, c)), c)


def relu(x, c):
    return activation(x, torch.nn.functional.relu, c)


def tanh(x, c):
    return activation(x, torch.tanh, c)


def rnn_step(x, h_prev, w_h, w_x, b, c):
    """
    Arguments:

        x: Input with shape (batch, input_dim)

        h_prev: Previous hidden state with shape (batch, hidden_dim)

        w_h: Weight matrix for hidden-hidden transition with shape (hidden_dim,
            hidden_dim). Note: This matrix lives in euclidian space

        w_x: Weight matrix for input-hidden transition with shape (input_dim,
            hidden_dim). Note: This matrix lives in euclidian space

        b: Bias with shape (hidden_dim,). Note: this lives in hyperbolic space

        c: c
    """
    hh = matmul(w_h, h_prev, c)
    xh = matmul(w_x, x, c)
    return add(add(hh, xh), b.unsqueeze(0))
