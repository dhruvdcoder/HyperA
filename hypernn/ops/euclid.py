import torch


def squared_distance(a, b):
    """Mobius a+b. dim(a)=dim(b)=batch,emb"""
    return torch.sum(a - b, dim=1, keepdim=True)
