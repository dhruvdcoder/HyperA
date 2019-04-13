import pytest
import hypernn.embeddings as embs
import hypernn.config as config
import hypernn.models as hmodels
from data.dummy import random_data_generator
from hypernn.training import train, default_params
import torch


def test_creation_HyperDeepAvgNet():
    hyp_emb_model = embs.load_word2vec(config.default_poincare_glove)
    hidden_dim = 50
    model = hmodels.HyperDeepAvgNet(hyp_emb_model, hidden_dim, 3)
    premise = torch.LongTensor([[1, 2]])
    hypo = torch.LongTensor([[3, 4]])
    t = model.forward((premise, hypo))


def test_training_HyperDeepAvgNet():
    params = default_params()
    hyp_emb_model = embs.load_word2vec(config.default_poincare_glove)
    hidden_dim = 50
    num_classes = 3
    batch_size = 64
    num_batches = 10
    seq_len = 20
    vocab_size = len(hyp_emb_model.vocab)
    model = hmodels.HyperDeepAvgNet(hyp_emb_model, hidden_dim, 3)

    train(model,
          random_data_generator(batch_size, seq_len, num_classes, num_batches),
          params)


if __name__ == '__main__':
    test_creation_HyperDeepAvgNet()
