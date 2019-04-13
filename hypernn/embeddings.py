""" Helpers for working with gensim/glove embeddings"""

import gensim


def load_word2vec(filepath):
    return gensim.models.KeyedVectors.load_word2vec_format(
        filepath, binary=False)
