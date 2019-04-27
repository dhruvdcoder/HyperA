"""Top level modules (models) for the project"""
import torch
import torch.nn as nn
import hypernn.modules as hnn
import hypernn.ops.mobius as m
import logging
logger = logging.getLogger(__name__)
default_c = m.default_c


class HyperDeepAvgNet(nn.Module):
    def __init__(self,
                 torchtext_vocab,
                 hidden_dim,
                 num_classes,
                 c,
                 freeze_emb=True,
                 emb_size=None,
                 init_avg_norm=None):
        super(HyperDeepAvgNet, self).__init__()
        logger.info("Creating HyperDeepAvgNet .. ")
        if torchtext_vocab.vectors is not None:
            self.emb_size = torchtext_vocab.vectors.size(1)
        else:
            self.emb_size = emb_size
        logger.info("Emb size: {}".format(self.emb_size))
        self.vocab_size = len(torchtext_vocab)
        logger.info("vocab_size :{}".format(self.vocab_size))
        self.hidden_dim = hidden_dim
        logger.info("hidden_dim :{}".format(self.hidden_dim))
        self.num_classes = num_classes
        self.c = c
        self.freeze_emb = freeze_emb
        if torchtext_vocab.vectors is not None:
            self.emb = hnn.HyperEmbeddings.from_torchtext_vocab(
                torchtext_vocab, self.c, sparse=False, freeze=freeze_emb)
        else:
            self.emb = hnn.HyperEmbeddings(
                self.vocab_size,
                self.emb_size,
                padding_idx=1,
                sparse=False,
                init_avg_norm=init_avg_norm)
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
                 rnn='RNN',
                 freeze_emb=False,
                 emb_size=None,
                 init_avg_norm=None):
        super(ConcatRNN, self).__init__()
        if torchtext_vocab.vectors is not None:
            self.emb_size = torchtext_vocab.vectors.size(1)
        else:
            self.emb_size = emb_size
        logger.info("Emb size: {}".format(self.emb_size))
        self.vocab_size = len(torchtext_vocab)
        logger.info("vocab_size :{}".format(self.vocab_size))
        self.hidden_dim = hidden_dim
        logger.info("hidden_dim :{}".format(self.hidden_dim))
        self.num_classes = num_classes
        self.freeze_emb = freeze_emb
        self.c = c
        if torchtext_vocab.vectors is not None:
            self.emb = hnn.HyperEmbeddings.from_torchtext_vocab(
                torchtext_vocab, self.c, sparse=False, freeze=freeze_emb)
        else:
            self.emb = hnn.HyperEmbeddings(
                self.vocab_size,
                self.emb_size,
                padding_idx=1,
                sparse=False,
                init_avg_norm=init_avg_norm)
        # Stacks the 2 matrices in the timestep dimension (NxWxV - W dimension)
        self.rnn_type = rnn
        logger.info("Using {} cell...".format(rnn))
        self.rnn = hnn._rnns[rnn](self.emb_size, self.hidden_dim)
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
            rolled_vector.size(0),
            self.hidden_dim,
            dtype=rolled_vector.dtype,
            device=rolled_vector.device)
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


class AddRNN(nn.Module):
    def __init__(self,
                 torchtext_vocab,
                 hidden_dim,
                 num_classes,
                 c,
                 rnn='RNN',
                 freeze_emb=False,
                 emb_size=None,
                 init_avg_norm=None):
        super(AddRNN, self).__init__()
        if torchtext_vocab.vectors is not None:
            self.emb_size = torchtext_vocab.vectors.size(1)
        else:
            self.emb_size = emb_size
        logger.info("Emb size: {}".format(self.emb_size))
        self.vocab_size = len(torchtext_vocab)
        logger.info("vocab_size :{}".format(self.vocab_size))
        self.hidden_dim = hidden_dim
        logger.info("hidden_dim :{}".format(self.hidden_dim))
        self.num_classes = num_classes
        self.freeze_emb = freeze_emb
        self.c = c
        if torchtext_vocab.vectors is not None:
            self.emb = hnn.HyperEmbeddings.from_torchtext_vocab(
                torchtext_vocab, self.c, sparse=False, freeze=freeze_emb)
        else:
            self.emb = hnn.HyperEmbeddings(
                self.vocab_size,
                self.emb_size,
                padding_idx=1,
                sparse=False,
                init_avg_norm=init_avg_norm)
        # Stacks the 2 matrices in the timestep dimension (NxWxV - W dimension)
        self.rnn_type = rnn
        logger.info("Using {} cell...".format(rnn))
        self.rnnp = hnn._rnns[rnn](self.emb_size, self.hidden_dim)
        self.rnnh = hnn._rnns[rnn](self.emb_size, self.hidden_dim)
        self.combinep = hnn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.combineh = hnn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.logits = hnn.Logits(hidden_dim, num_classes, c=c)

    def forward(self, inp):
        premise, hypothesis = inp

        # Project to Hyperbolic embedding space
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        batch_size = premise_emb.size(0)
        h0p = torch.zeros(
            batch_size,
            self.hidden_dim,
            dtype=premise_emb.dtype,
            device=premise_emb.device)
        h0h = torch.zeros(
            batch_size,
            self.hidden_dim,
            dtype=hypothesis_emb.dtype,
            device=hypothesis_emb.device)
        outputp = self.rnn((premise_emb, h0p))
        outputh = self.rnn((hypothesis_emb, h0h))
        p_rep = self.combinep(outputp)
        h_rep = self.combinep(outputh)
        rep = m.add(p_rep, h_rep, self.c)
        logits = self.logits(rep)
        return logits

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        for layer in [
                self.rnnp, self.rnnh, self.combinep, self.combineh, self.logits
        ]:
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
        for layer in [
                self.rnnp, self.rnnh, self.combinep, self.combineh, self.logits
        ]:
            params = layer.get_hyperbolic_params()
            bias_params += params
        hyp_params.append({'params': bias_params, 'lr': bias_lr})
        return hyp_params


model_zoo = {
    'hconcatrnn': ConcatRNN,
    'hdeepavg': HyperDeepAvgNet,
    'haddrnn': AddRNN
}
