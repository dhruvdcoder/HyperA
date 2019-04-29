""" Contains hypernn layers as pytorch modules"""

import torch
import torch.nn as nn
import hypernn.ops.mobius as m
import math


class Linear(nn.Module):
    """Hyperbolic linear transformation layer"""

    def __init__(self, in_features, out_features, bias=True, c=m.default_c):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5), mode='fan_out')
        # mode is fan_out because out weight is transpose of the weight of usual
        # linear layer
        if self.bias is not None:
            #fan_in = self.weight.size(0)
            #bound = 1 / math.sqrt(fan_in)
            #nn.init.uniform_(self.bias, -bound, bound)
            nn.init.zeros_(self.bias)

    def forward(self, inp):
        out = m.matmul(self.weight, inp, self.c)
        if self.bias is not None:
            out = m.add(out, self.bias.unsqueeze(0), self.c)

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def get_euclidean_params(self):
        """Convenience function to collect params for optmization"""
        return [self.weight]

    def get_hyperbolic_params(self):
        if self.bias is None:
            return []
        else:
            return [self.bias]


class SpecialLinear(Linear):
    """ Designed to apply W.M * x ++ b
    (.) being euclidean matmul (*) being hyperbolic matmul
    (++) being mobius add
    """

    def forward(self, M, inp):
        assert M.size(-1) == self.weight.size(0)
        assert M.size(-1) == M.size(-2)  # Should be square
        mat = torch.matmul(M, self.weight)
        out = m.matmul(mat, inp, self.c)
        if self.bias is not None:
            out = m.add(out, self.bias.unsqueeze(0), self.c)
        return out


activations_dict = {'tanh': m.tanh, 'relu': m.relu, 'id': m.id}


class Dense(Linear):
    """Hyperbolic Linear transformation followed by
    a non-linearity"""

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 activation='id',
                 c=m.default_c):
        super(Dense, self).__init__(in_features, out_features, bias, c)
        self.activation = activation

    def forward(self, inp):
        after_linear = super().forward(inp)
        res = activations_dict[self.activation](after_linear, self.c)
        return res

    def extra_repr(self):
        return ('in_features={}, out_features={}, bias={},'
                ' activation={}').format(self.in_features, self.out_features,
                                         self.bias is not None,
                                         self.activation)


class Logits(nn.Module):
    """Logits in hyperbolic space

    Contains trainable parameters 'p' (shape=out_features, in_features)
    and 'a' (shape same as p). They can be accessed by the same name.
    **Note:** 'a' is a euclidian parameter while p is in hyperbolic space.
    This has to be kept in mind while calculating the grads.
    """

    def __init__(self, in_features, out_features, c=m.default_c):
        """

        Arguments:

            in_features: Hidden dim size of previous layer

            out_features: Number of classes

            c: c
        """
        super(Logits, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.c = c
        self.p = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.a)
        nn.init.zeros_(self.p)

    def forward(self, inp):
        return m.logits(inp, self.p, self.a, self.c)

    def extra_repr(self):
        return ('in_features (num_hidden_dim)={}, out_features(num_classes)={}'
                ).format(self.in_features, self.out_features)

    def get_hyperbolic_params(self):
        """Convenience function to collect params for optmization"""
        return [self.p]

    def get_euclidean_params(self):
        return [self.a]


class HyperEmbeddings(nn.Embedding):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
            _weight=None,
            init_avg_norm=0.001,
    ):
        self.init_avg_norm = init_avg_norm
        super(HyperEmbeddings, self).__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight)

    def get_hyperbolic_params(self):
        wts = self.weight
        return [wts]

    def get_euclidean_params(self):
        return []

    def reset_parameters(self):
        maxval = (3. * (self.init_avg_norm**2) / (2. * self.embedding_dim))**(
            1. / 3)
        torch.nn.init.uniform_(self.weight, -maxval, maxval)
        # IMPORTANT set padding emb to zero
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    @classmethod
    def from_gensim_model(cls, gensim_model, c, freeze=False, sparse=False):
        emb_tensor = torch.tensor(gensim_model.vectors)
        return super(HyperEmbeddings, cls).from_pretrained(
            emb_tensor, freeze=freeze, sparse=sparse)

    @classmethod
    def from_vectors(cls, vectors, c, freeze=False, sparse=False):
        return super(HyperEmbeddings, cls).from_pretrained(
            vectors, freeze=freeze, sparse=sparse)

    @classmethod
    def from_torchtext_vocab(cls, vocab, c, freeze=False, sparse=False):
        return super(HyperEmbeddings, cls).from_pretrained(
            vocab.vectors, freeze=freeze, sparse=sparse)


class HyperRNNCell(nn.Module):
    """TODO: support num_layers parameter"""

    def __init__(self, input_size, hidden_dim, activation='id', c=m.default_c):
        super(HyperRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.l_ih = Linear(input_size, hidden_dim)
        self.l_hh = Linear(hidden_dim, hidden_dim)
        self.c = c
        self.activation = activation

    def forward(self, inp_tuple):
        inp, prev_h = inp_tuple
        # TODO: Bias?
        # h_next = m.tanh(m.add(self.l_premise_hyp(inp), self.l_hidden(prev_h)), c=self.c)
        # print (inp.size())
        prem = self.l_ih(inp)
        hid = self.l_hh(prev_h)
        h_next = activations_dict[self.activation](
            m.add(prem, hid, self.c), c=self.c)
        return h_next

    def get_hyperbolic_params(self, bias_lr=0.01):
        """Get list of hyperbolic params"""
        bias_params = []
        for layer in [self.l_ih, self.l_hh]:
            params = layer.get_hyperbolic_params()
            bias_params += params
        return bias_params

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        for layer in [self.l_ih, self.l_hh]:
            params = layer.get_euclidean_params()
            params_list += params
        return params_list

class HyperRNN(nn.Module):
    def __init__(self, input_size, hidden_size, activation='id',
                 c=m.default_c):
        super(HyperRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.c = c
        self.rnn_cell = HyperRNNCell(
            self.input_size, self.hidden_size, activation=activation, c=self.c)

    def forward(self, inp):
        x, h0 = inp
        #x = inp
        #h0 = torch.zeros(x.size(0), self.hidden_size).double()
        tsteps = x.size(-2)
        prev_h = h0
        for t in range(tsteps):
            inp_cell = x[:, t, :]
            prev_h = self.rnn_cell((inp_cell, prev_h))

        return prev_h

    def get_hyperbolic_params(self, emb_lr=0.1, bias_lr=0.01):
        """Get list of hyperbolic params"""
        return self.rnn_cell.get_hyperbolic_params()

    def get_euclidean_params(self, lr=0.001):
        return self.rnn_cell.get_euclidean_params()

class HyperGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, activation='id', c=m.default_c):
        super(HyperGRUCell, self).__init__()
        self.l_inp_z = Linear(input_size, hidden_size, bias=False)
        self.l_hid_z = Linear(hidden_size, hidden_size)

        self.l_inp_r = Linear(input_size, hidden_size, bias=False)
        self.l_hid_r = Linear(hidden_size, hidden_size)

        self.l_inp_h = Linear(input_size, hidden_size, bias=False)
        self.l_hid_h = Linear(hidden_size, hidden_size)

        self.activation = activation
        self.c = c

    def forward(self, inp_tuple):
        inp, prev_h = inp_tuple

        # GRU implementation following the wildml blog
        # http://www.wildml.com/2015/10/
        # recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/

        # z = m.sigmoid(m.add(self.l_inp_z(inp), self.l_hid_z(prev_h), c=self.c), c=self.c)
        # r = m.sigmoid(m.add(self.l_inp_r(inp), self.l_hid_r(prev_h), c=self.c), c=self.c)

        z = m.sigmoid_hyp_to_eucl(m.add(self.l_inp_z(inp), self.l_hid_z(prev_h), c=self.c), c=self.c)
        r = m.sigmoid_hyp_to_eucl(m.add(self.l_inp_r(inp), self.l_hid_r(prev_h), c=self.c), c=self.c)
        temp_activ = activations_dict[self.activation](
            m.add(m.pointwise_prod(self.l_hid_h(prev_h), r, c=self.c), self.l_inp_h(inp), c=self.c),
                c=self.c)

        # There are 2 version of final state calculation, based on whether you want to use
        # (1 - z) on previous state or h'
        # They are:
        # (1 - z) * h' + z * st-1 = z * (st-1 - h) + h'
        # (1 - z) * st-1 + z * h' = z * (h' - st-1) + st-1

        # We used the second version

        minus_h = m.add(-prev_h, temp_activ, c=self.c)
        h_next = m.add(prev_h, m.pointwise_prod(minus_h, z, c=self.c), c=self.c)
        # print (h_next.size())
        return h_next

    def get_hyperbolic_params(self, bias_lr=0.01):
        """Get list of hyperbolic params"""
        bias_params = []
        for layer in [self.l_inp_z, self.l_hid_z, self.l_inp_r, self.l_hid_r, self.l_inp_h, self.l_hid_h]:
            params = layer.get_hyperbolic_params()
            bias_params += params
        return bias_params

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        for layer in [self.l_inp_z, self.l_hid_z, self.l_inp_r, self.l_hid_r, self.l_inp_h, self.l_hid_h]:
            params = layer.get_euclidean_params()
            params_list += params
        return params_list

class HyperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, activation='id', c=m.default_c):
        super(HyperGRU, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.c = c
        self.gru_cell = HyperGRUCell(hidden_size, input_size, activation=activation, c=self.c)

    def forward(self, inp):
        # Assert that inp.dimension is of form (NxWxE)
        # assert (inp)
        # h0 = torch.zeros(inp.size()[0], self.hidden_size).double()

        x, h0 = inp
        tsteps = x.size()[-2]
        prev_h = h0
        for t in range(tsteps):
            inp_cell = x[:, t, :]
            next_h = self.gru_cell((inp_cell, prev_h))
            prev_h = next_h

        return next_h

    def get_hyperbolic_params(self, emb_lr=0.1, bias_lr=0.01):
        """Get list of hyperbolic params"""
        return self.gru_cell.get_hyperbolic_params()

    def get_euclidean_params(self, lr=0.001):
        return self.gru_cell.get_euclidean_params()


_rnns = {'RNN': HyperRNN, 'GRU': HyperGRU}
