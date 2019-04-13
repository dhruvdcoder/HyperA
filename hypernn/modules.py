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
            fan_in = self.weight.size(0)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):
        out = m.matmul(self.weight, inp, self.c)
        if self.bias is not None:
            out = m.add(out, self.bias.unsqueeze(0), self.c)

        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def get_hyperbolic_params(self):
        """Convenience function to collect params for optmization"""
        return [self.weight]

    def get_euclidean_params(self):
        if self.bias is None:
            return []
        else:
            return [self.bias]


class Dense(Linear):
    """Hyperbolic Linear transformation followed by
    a non-linearity"""

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 activation=m.tanh,
                 c=m.default_c):
        super(Dense, self).__init__(in_features, out_features, bias, c)
        self.activation = lambda x: activation(x, c=self.c)

    def forward(self, inp):
        after_linear = super().forward(inp)
        return self.activation(after_linear)

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
    def get_hyperbolic_params(self):
        wts = self.weight
        return [wts]

    def get_euclidean_params(self):
        return []

    @classmethod
    def from_gensim_model(cls,
                          gensim_model,
                          freeze=False,
                          sparse=True,
                          c=m.default_c):
        emb_tensor = torch.tensor(gensim_model.vectors)
        return super(HyperEmbeddings, cls).from_pretrained(
            emb_tensor, freeze=freeze, sparse=sparse)

class HyperRNNCell(nn.Module):
    def __init__(self, hidden_dim, emb_size, c=m.default_c):
        super(HyperRNNCell, self).__init__()
        self.l_premise_hypoth = Linear(emb_size, hidden_dim, bias=False)
        self.l_hidden = Linear(hidden_dim, hidden_dim)
        self.c = c

    def forward(self, inp_tuple):
        inp, prev_h = inp_tuple
        # TODO: Bias?
        # h_next = m.tanh(m.add(self.l_premise_hyp(inp), self.l_hidden(prev_h)), c=self.c)
        # print (inp.size())
        prem = self.l_premise_hypoth(inp)
        hid = self.l_hidden(prev_h)
        h_next = m.tanh(m.add(prem, hid), c=self.c)
        return h_next


class HyperRNN(nn.Module):
    def __init__(self, hidden_dim, input_dims, c=m.default_c):
        super(HyperRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dims = input_dims
        self.c = c
        self.RNNCell = HyperRNNCell(self.hidden_dim, input_dims, self.c)

    def forward(self, inp):
        # Assert that inp.dimension is of form (NxWxE)
        # assert (inp)

        h0 = torch.zeros(inp.size()[0], self.hidden_dim).double()
        tsteps = inp.size()[-2]
        prev_h = h0
        for t in range(tsteps):
            # print ("Executing timestep: ", t)
            inp_cell = inp[:, t, :]
            # print (inp_cell.size())
            next_h = self.RNNCell((inp_cell, prev_h))

        # next_h = self.RNNCell((inp, prev_h))
        return next_h


    def get_hyperbolic_params(self, emb_lr=0.1, bias_lr=0.01):
        """Get list of hyperbolic params"""
        hyp_params = []
        hyp_params.append({
            'params': self.emb.get_hyperbolic_params(),
            'lr': emb_lr
        })
        bias_params = [
            layer.get_hyperbolic_params() for layer in [
                self.dense_premise, self.dense_hypothesis, self.dense_combine,
                self.logits
            ]
        ]
        hyp_params.append({'params': bias_params, 'lr': bias_lr})

    def get_euclidean_params(self, lr=0.001):
        params_list = [
            layer.get_euclidean_params() for layer in [
                self.dense_premise, self.dense_hypothesis, self.dense_combine,
                self.logits
            ]
        ]
        euc_params = [{'params': params_list, 'lr': lr}]
        return euc_params
