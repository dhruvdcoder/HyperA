"""Top level modules (models) for the project"""
import torch
import torch.nn as nn
import hypernn.modules as hnn
import hypernn.ops.mobius as m


class HyperDeepAvgNet(nn.Module):
    def __init__(self, gensim_emb, hidden_dim, num_classes, c):
        super(HyperDeepAvgNet, self).__init__()
        self.emb_size = gensim_emb.vector_size
        self.vocab_size = len(gensim_emb.vocab)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.c = c
        self.emb = hnn.HyperEmbeddings.from_gensim_model(
            gensim_emb, sparse=False)
        self.dense_premise = hnn.Dense(self.emb_size, hidden_dim, c=c)
        self.dense_hypothesis = hnn.Dense(self.emb_size, hidden_dim, c=c)
        self.dense_combine = hnn.Dense(
            2 * self.hidden_dim, self.hidden_dim, c=c)
        self.logits = hnn.Logits(hidden_dim, num_classes, c=c)
        self.avg = lambda x: m.mean(x, c, dim=-2)
        self.cat = lambda premise, hypothesis: torch.cat((premise, hypothesis), -1)

    def forward(self, inp):
        premise, hypothesis = inp
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        premise_rep = self.dense_premise(self.avg(premise_emb))
        hypothesis_rep = self.dense_hypothesis(self.avg(hypothesis_emb))
        concat_rep = self.cat(premise_rep, hypothesis_rep)
        logits = self.logits(self.dense_combine(concat_rep))
        return logits

    def get_hyperbolic_params(self, emb_lr=0.1, bias_lr=0.01):
        """Get list of hyperbolic params"""
        hyp_params = []
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
    def __init__(self, gensim_emb, hidden_dim, num_classes, c):
        super(ConcatRNN, self).__init__()
        self.emb_size = gensim_emb.vector_size
        self.vocab_size = len(gensim_emb.vocab)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.c = c
        self.emb = hnn.HyperEmbeddings.from_gensim_model(
            gensim_emb, sparse=False)

        # Stacks the 2 matrices in the timestep dimension (NxWxV - W dimension)
        self.cat = lambda premise, hypothesis: torch.cat((premise, hypothesis), -2)
        self.rnn = hnn.HyperRNN(self.emb_size, self.hidden_dim)
        self.logits = hnn.Logits(hidden_dim, num_classes, c=c)

    def forward(self, inp):
        premise, hypothesis = inp

        # Project to Hyperbolic embedding space
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        rolled_vector = self.cat(premise_emb, hypothesis_emb)
        h0 = torch.zeros(rolled_vector.size(0), self.hidden_dim)
        output = self.rnn((rolled_vector, h0))
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
