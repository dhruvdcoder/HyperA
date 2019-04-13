"""Top level modules (models) for the project"""
import torch
import torch.nn as nn
import hypernn.modules as hnn
import hypernn.ops.mobius as m


class HyperDeepAvgNet(nn.Module):
    def __init__(self, gensim_emb, hidden_dim, num_classes, c=m.default_c):
        super(HyperDeepAvgNet, self).__init__()
        self.emb_size = gensim_emb.vector_size
        self.vocab_size = len(gensim_emb.vocab)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.c = c
        self.emb = hnn.HyperEmbeddings.from_gensim_model(gensim_emb)
        self.dense_premise = hnn.Dense(self.emb_size, hidden_dim, c=c)
        self.dense_hypothesis = hnn.Dense(self.emb_size, hidden_dim, c=c)
        self.dense_combine = hnn.Dense(
            2 * self.hidden_dim, self.hidden_dim, c=c)
        self.logits = hnn.Logits(hidden_dim, num_classes, c=c)
        self.avg = lambda x: torch.mean(x, dim=-2)
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
