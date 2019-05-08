import torch


def pick_along_seq(t, indices, keepdims=False):
    """ Moves along the batch dimension, picks
    the element specified by the corresponding
    entry in indices

    Arguments:
        
        t: Tensor of shape (batch, seq, hidden_dim)

        indices: LongTensor of shape (batch,). ith
            element specifies the element to pick in
            the sequence of ith example in the batch

    Returns:
        tensor of shape (batch, hidden_dim) if keepdims is
        False else (batch, 1, hidden_dim)
    """
    assert len(t.shape) == 3
    assert len(indices.shape) == 1
    assert indices.size(0) == t.size(0)
    hidden_dim = t.size(2)
    res = torch.gather(
        t, 1,
        indices.view(-1, 1).unsqueeze(2).repeat(1, 1, hidden_dim))
    if keepdims:
        res = res.squeeze(1)
    return res
