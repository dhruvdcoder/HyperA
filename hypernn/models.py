"""Top level modules (models) for the project"""
import torch
import torch.nn as nn
import hypernn.modules as hnn
import hypernn.ops.mobius as m
import logging
from ops.common import pick_along_seq
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
                 init_avg_norm=None,
                 **kwargs):
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
        (premise, p_sent_len), (hypothesis, h_sent_len) = inp
        premise_emb = self.emb(premise)
        hypothesis_emb = self.emb(hypothesis)
        premise_rep = self.dense_premise(
            m.mean(
                premise_emb, self.c, dim=-2,
                to_divide=p_sent_len.unsqueeze(1)))
        hypothesis_rep = self.dense_hypothesis(
            m.mean(
                hypothesis_emb,
                self.c,
                dim=-2,
                to_divide=h_sent_len.unsqueeze(1)))
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
        output = self.rnn((rolled_vector, h0))[-1]
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
                 init_avg_norm=None,
                 combine_op='add',
                 **kwargs):
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
        self.combine_op = combine_op
        if combine_op == 'add':
            logits_input_dim = hidden_dim
        elif combine_op == 'concat':
            logits_input_dim = 2 * hidden_dim
        else:
            raise ValueError("Invalid combine_op")
        self.logits = hnn.Logits(logits_input_dim, num_classes, c=c)

    def _add_combine(self, p, h):
        return m.add(p, h, self.c)

    def _concat_combine(self, p, h):
        return torch.cat((p, h), -1)

    def combine(self, p, h):
        if self.combine_op == 'add':
            rep = self._add_combine(p, h)
        else:
            rep = self._concat_combine(p, h)
        return rep

    def forward(self, inp):
        (premise, p_sent_len), (hypothesis, h_sent_len) = inp

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
        #outputp = self.rnnp((premise_emb, h0p))[:, -1, :]
        outputp = pick_along_seq(
            self.rnnp((premise_emb, h0p)),
            p_sent_len - 1)  # keep this -1 if using <eos>
        #outputh = self.rnnh((hypothesis_emb, h0h))[:, -1, :]
        outputh = pick_along_seq(
            self.rnnh((hypothesis_emb, h0h)),
            h_sent_len - 1)  # keep this -1 if using <eos>
        p_rep = self.combinep(outputp)
        h_rep = self.combineh(outputh)
        rep = self.combine(p_rep, h_rep)
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


class ConcatGRU(nn.Module):
    def __init__(self,
                 torchtext_vocab,
                 hidden_dim,
                 num_classes,
                 c,
                 freeze_emb=False,
                 emb_size=None,
                 init_avg_norm=None):
        super(ConcatGRU, self).__init__()
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
        self.gru = hnn.HyperGRU(self.emb_size, self.hidden_dim)
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
        output = self.gru((rolled_vector,
                           h0))[:, -1, :]  # take the final state only
        #output = self.gru(rolled_vector)
        logits = self.logits(output)
        return logits

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        for layer in [self.gru, self.logits]:
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
        for layer in [self.gru, self.logits]:
            params = layer.get_hyperbolic_params()
            bias_params += params
        hyp_params.append({'params': bias_params, 'lr': bias_lr})
        return hyp_params


class HyperDeepAvgNetAttn(nn.Module):
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
        self.const_bias_param = torch.nn.Parameter(torch.Tensor(hidden_dim))
        torch.nn.init.zeros_(self.const_bias_param)
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
        batch_size = hypothesis_rep.size(0)
        seq_len = hypothesis_rep.size(1)
        const = torch.repeat(batch_size, seq_len, 1)
        values = torch.cat((hypothesis_emb, const), -1)
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


class AddRNNAttn(nn.Module):
    def __init__(self,
                 torchtext_vocab,
                 hidden_dim,
                 num_classes,
                 c,
                 rnn='RNN',
                 freeze_emb=False,
                 emb_size=None,
                 init_avg_norm=None):
        super(AddRNNAttn, self).__init__()
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
        self.const_bias_param = torch.nn.Parameter(torch.Tensor(hidden_dim))
        torch.nn.init.zeros_(self.const_bias_param)
        #self.combinep = hnn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        #self.combineh = hnn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.logits = hnn.Logits(2 * hidden_dim, num_classes, c=c)

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
        outputp = self.rnnp((premise_emb, h0p))
        outputh = self.rnnh((hypothesis_emb, h0h))[:-1:]
        batch_size = outputp.size(0)
        seq_len = outputp.size(1)
        const = self.const_bias_param.repeat(batch_size, seq_len, 1)
        values = torch.cat((outputp, const), -1)
        out = m.single_query_attn(outputp, outputh, values, self.c)
        rep = m.mean(out, self.c, dim=-2)
        logits = self.logits(rep)
        return logits

    def get_euclidean_params(self, lr=0.001):
        params_list = []
        # self.combinep, self.combineh,
        for layer in [self.rnnp, self.rnnh, self.logits]:
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
        bias_params = [self.const_bias_param]
        #self.combinep, self.combineh,
        for layer in [self.rnnp, self.rnnh, self.logits]:
            params = layer.get_hyperbolic_params()
            bias_params += params
        hyp_params.append({'params': bias_params, 'lr': bias_lr})
        return hyp_params


model_zoo = {
    'hconcatrnn': ConcatRNN,
    'hdeepavg': HyperDeepAvgNet,
    'haddrnn': AddRNN,
    'hconcatgru': ConcatGRU,
    'addrnnattn': AddRNNAttn
}
