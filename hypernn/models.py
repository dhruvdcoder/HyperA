"""Top level modules (models) for the project"""
import torch
import torch.nn as nn
import hypernn.modules as hnn
import hypernn.ops.mobius as m
default_c = m.default_c


class HyperDeepAvgNet(nn.Module):
    def __init__(self,
                 torchtext_vocab,
                 hidden_dim,
                 num_classes,
                 c,
                 freeze_emb=True):
        super(HyperDeepAvgNet, self).__init__()
        self.emb_size = torchtext_vocab.vectors.size(1)
        self.vocab_size = len(torchtext_vocab)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.c = c
        self.freeze_emb = freeze_emb
        self.emb = hnn.HyperEmbeddings.from_torchtext_vocab(
            torchtext_vocab, self.c, sparse=False, freeze=freeze_emb)
        self.dense_premise = hnn.Dense(self.emb_size, hidden_dim, c=c)
        self.dense_hypothesis = hnn.Dense(self.emb_size, hidden_dim, c=c)
        self.dense_combine = hnn.Dense(
            2 * self.hidden_dim, self.hidden_dim, c=c)
        self.logits = hnn.Logits(hidden_dim, num_classes, c=c)

    def forward(self, inp):
        premise, hypothesis = inp
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        premise_rep = self.dense_premise(m.mean(premise_emb, self.c, dim=-2))
        hypothesis_rep = self.dense_hypothesis(
            m.mean(hypothesis_emb, self.c, dim=-2))
        concat_rep = torch.cat((premise_rep, hypothesis_rep), -1)
        logits = self.logits(self.dense_combine(concat_rep))
        return logits

    def get_hyperbolic_params(self, emb_lr=0.1, bias_lr=0.01):
        """Get list of hyperbolic params"""
        hyp_params = []
        if self.emb.weight.requires_grad:
            hyp_params.append({
                'params': self.emb.get_hyperbolic_params(),
                'lr': emb_lr
            })
        bias_params = []
        for layer in [
                self.dense_premise, self.dense_hypothesis, self.dense_combine,
                self.logits
        ]:
            params = layer.get_hyperbolic_params()
            bias_params += params

        hyp_params.append({'params': bias_params, 'lr': bias_lr})
        return hyp_params

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        for layer in [
                self.dense_premise, self.dense_hypothesis, self.dense_combine,
                self.logits
        ]:
            params = layer.get_euclidean_params()
            params_list += params

        euc_params = [{'params': params_list, 'lr': lr}]
        return euc_params


class ConcatRNN(nn.Module):
    def __init__(self,
                 torchtext_vocab,
                 hidden_dim,
                 num_classes,
                 c,
                 freeze_emb=False):
        super(ConcatRNN, self).__init__()
        self.emb_size = torchtext_vocab.vectors.size(1)
        self.vocab_size = len(torchtext_vocab)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.freeze_emb = freeze_emb
        self.c = c
        self.emb = hnn.HyperEmbeddings.from_torchtext_vocab(
            torchtext_vocab, self.c, sparse=False, freeze=self.freeze_emb)

        # Stacks the 2 matrices in the timestep dimension (NxWxV - W dimension)
        self.rnn = hnn.HyperRNN(self.emb_size, self.hidden_dim)
        self.logits = hnn.Logits(hidden_dim, num_classes, c=c)

    def forward(self, inp):
        premise, hypothesis = inp

        # Project to Hyperbolic embedding space
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        #rolled_vector = self.cat(premise_emb, hypothesis_emb)
        rolled_vector = torch.cat((premise_emb, hypothesis_emb), -2)
        #h0 = torch.zeros(rolled_vector.size(0), self.hidden_dim).double()
        h0 = torch.zeros(
            rolled_vector.size(0), self.hidden_dim, dtype=rolled_vector.dtype)
        output = self.rnn((rolled_vector, h0))
        #output = self.rnn(rolled_vector)
        logits = self.logits(output)
        return logits

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        for layer in [self.rnn, self.logits]:
            params = layer.get_euclidean_params()
            params_list += params

        euc_params = [{'params': params_list, 'lr': lr}]
        return euc_params

    def get_hyperbolic_params(self, emb_lr=0.1, bias_lr=0.01):
        """Get list of hyperbolic params"""
        hyp_params = []
        if self.emb.weight.requires_grad:
            hyp_params.append({
                'params': self.emb.get_hyperbolic_params(),
                'lr': emb_lr
            })
        bias_params = []
        for layer in [self.rnn, self.logits]:
            params = layer.get_hyperbolic_params()
            bias_params += params
        hyp_params.append({'params': bias_params, 'lr': bias_lr})
        return hyp_params
