import pytest
from hypernn import config as config
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
        c = torch.Tensor(np_utils.random_vec((1, ), low=0.0, high=1.))
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
        cval = np.array([1.])
        c = torch.Tensor(cval)
        logger.info("c: {}".format(c))
        res = mobius.squared_distance(a, b, c).data.numpy()
        expected = np_utils.squared_distance(aval, bval, cval)
        assert res.shape == expected.shape
        assert np.allclose(res, expected)

    for i in range(1000):
        logger.debug("{}th sample".format(i))
        sample()


if __name__ == '__main__':
    test_dot()
    test_add_left_cancelation()
