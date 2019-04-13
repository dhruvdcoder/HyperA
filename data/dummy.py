""" Function to generate dummy datasets"""
import numpy as np
import torch


def random_data(num_samples, seq_len, num_classes, vocab_size):
    premises = np.random.randint(0, vocab_size - 1, (num_samples, seq_len))
    hypo = np.random.randint(0, vocab_size - 1, (num_samples, seq_len))
    labels = np.random.randint(0, num_classes - 1, num_samples)
    inp = (torch.LongTensor(premises), torch.LongTensor(hypo))
    labels = torch.LongTensor(labels)
    return (inp, labels)


def random_data_generator(batch_size,
                          seq_len,
                          num_classes,
                          vocab_size,
                          num_batches=10):
    def gen():
        for i in range(num_batches):
            yield random_data(batch_size, seq_len, num_classes, vocab_size)

    return gen
