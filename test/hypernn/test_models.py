import pytest
import hypernn.embeddings as embs
import hypernn.config as config
import hypernn.models as hmodels
import torch


def test_creation_HyperDeepAvgNet():
    hyp_emb_model = embs.load_word2vec(config.default_poincare_glove)
    hidden_dim = 50
    model = hmodels.HyperDeepAvgNet(hyp_emb_model, hidden_dim, 3)
    premise = torch.LongTensor([[1, 2]])
    hypo = torch.LongTensor([[3, 4]])
    t = model.forward((premise, hypo))


if __name__ == '__main__':
    test_creation_HyperDeepAvgNet()
