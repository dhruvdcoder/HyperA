import pytest
import hypernn.modules as hnn
import test.np_utils as np_utils
import torch
import numpy as np
import logging
logger = logging.getLogger(__name__)


def test_Linear_forward():
    in_features = 100
    out_features = 50
    batch_size = 64

    def test_case():
        #c_val = np_utils.random_vec((1, ), low=0.5, high=1.)
        c_val = np.array([1.])
        logger.debug("c={}".format(c_val))
        c = torch.Tensor(c_val)
        x = torch.Tensor(np_utils.random_vec((batch_size, in_features)))
        hnn_linear = hnn.Linear(in_features, out_features, c=c)
        M = hnn_linear.weight.data.numpy()
        b = hnn_linear.bias.data.numpy()
        np_res = np_utils.Linear(x.data.numpy(), M, b, c_val)
        torch_res = hnn_linear(x)
        assert np.allclose(np_res, torch_res.data.numpy(), atol=1e-7)

    for i in range(5000):
        logger.debug(i)
        test_case()


def test_Linear_grad():
    in_features = 100
    out_features = 50
    batch_size = 64

    c_val = np.array([1.])
    c = torch.Tensor(c_val).double()
    hnn_linear = hnn.Linear(in_features, out_features, bias=True, c=c).double()

    def test_case():
        x = torch.tensor(
            np_utils.random_vec((batch_size, in_features)),
            requires_grad=True).double()
        torch.autograd.gradcheck(hnn_linear, x)

    for i in range(10):
        test_case()


def test_Logits_grad():
    emb_size = 100
    K = 5
    batch_size = 64

    c_val = np.array([1.])
    c = torch.Tensor(c_val).double()
    hnn_logits = hnn.Logits(emb_size, K, c=c).double()

    def test_case():
        x = torch.tensor(
            np_utils.random_vec((batch_size, emb_size)),
            requires_grad=True).double()
        torch.autograd.gradcheck(hnn_logits, x)

    for i in range(10):
        test_case()


if __name__ == '__main__':
    #test_Linear_forward()
    test_Linear_grad()
