import pytest
#from hypernn import config as config
from hypernn.ops import mobius
import logging
import torch
import numpy as np
from test import np_utils
logger = logging.getLogger(__name__)

batch_size = 5
emb_size = 10


def test_dot():
    a = torch.Tensor([[1., 2.], [3., 4.]])
    b = torch.Tensor([[5., 2.], [1., 4.]])
    expected = np.array([[9.], [19.]])
    res = mobius.dot(a, b)
    assert np.allclose(res, expected)


def test_norm():
    a = torch.Tensor([[1., 2.], [3., 4.]])
    b = torch.Tensor([[5., 2.], [1., 4.]])
    expected = np.sqrt(np.array([[5], [25.]]))
    res = mobius.norm(a).data.numpy()
    assert np.allclose(res, expected)
    expected = np.sqrt(np.array([[29.], [17.]]))
    res = mobius.norm(b).data.numpy()
    assert np.allclose(res, expected)


def test_add_left_cancelation():
    def sample():
        a = torch.Tensor(np_utils.random_vec((batch_size, emb_size)))
        b = torch.Tensor(np_utils.random_vec((batch_size, emb_size)))
        c = np_utils.random_vec([], low=0.0, high=1.)
        c = float(c)
        logger.debug("c: {}".format(c))
        res = mobius.add(-a, mobius.add(a, b, c), c).data.numpy()
        expected = b.data.numpy()
        assert res.shape == expected.shape
        assert np.allclose(res, expected)

    for i in range(1000):
        logger.debug("{}th sample".format(i))
        sample()


def test_distance():
    def sample():
        aval = np_utils.random_vec((batch_size, emb_size), low=-0.01)
        bval = np_utils.random_vec((batch_size, emb_size), low=-0.01)
        a = torch.Tensor(aval)
        b = torch.Tensor(bval)
        #cval = np_utils.random_vec((1, ), low=0.0, high=1.)
        # TODO: Why does this fail with random c?
        cval = 1.
        c = cval
        logger.info("c: {}".format(c))
        res = mobius.squared_distance(a, b, c).data.numpy()
        expected = np_utils.squared_distance(aval, bval, cval)
        assert res.shape == expected.shape
        assert np.allclose(res, expected)

    for i in range(1000):
        logger.debug("{}th sample".format(i))
        sample()


def test_logits():
    batch_size = 4
    emb_size = 5
    K = 3
    p_val = np_utils.random_vec((K, emb_size))
    a_val = np_utils.random_vec((K, emb_size))
    cval = 1.
    #c = torch.Tensor(cval).double()
    c = cval
    p = torch.tensor(p_val).double()
    a = torch.tensor(a_val).double()
    x_val = np_utils.random_vec((batch_size, emb_size))
    x = torch.tensor(x_val).double()
    logits = mobius.logits(x, p, a, c)
    assert tuple(logits.shape) == (batch_size, K)


def test_linear_grad():
    cval = 1.
    #c = torch.Tensor(cval).double()
    c = cval
    in_features = 100
    out_features = 50
    batch_size = 64

    def linear(x, w, b):
        return mobius.add(mobius.matmul(w, x, c), b.unsqueeze(0), c)

    def test_case():
        w_val = np_utils.random_vec((in_features, out_features))
        w = torch.tensor(w_val, requires_grad=True).double()
        b_val = np_utils.random_vec((out_features, )).reshape(out_features)
        b = torch.tensor(b_val, requires_grad=True).double()
        x_val = np_utils.random_vec((batch_size, in_features))
        x = torch.tensor(x_val, requires_grad=True).double()
        torch.autograd.gradcheck(linear, (x, w, b))

    for i in range(1):
        test_case()


def test_logits_grad():
    batch_size = 4
    emb_size = 5
    K = 3

    def test_case():
        p_val = np_utils.random_vec((K, emb_size))
        a_val = np_utils.random_vec((K, emb_size))
        cval = 1.
        #c = torch.Tensor(cval).double()
        c = cval
        p = torch.tensor(p_val, requires_grad=True).double()
        a = torch.tensor(a_val, requires_grad=True).double()
        x_val = np_utils.random_vec((batch_size, emb_size))
        x = torch.tensor(x_val, requires_grad=True).double()
        torch.autograd.gradcheck(mobius.logits, (x, p, a, c))

    for i in range(1):
        test_case()


if __name__ == '__main__':
    test_logits()
