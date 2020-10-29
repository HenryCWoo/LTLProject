import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Attention(nn.Module):
    def __init__(self, vec_dim, d_k=64):
        super().__init__()

        self.vec_dim = vec_dim
        self.d_k = d_k

        self.w_q = nn.Linear(vec_dim, d_k)
        self.w_k = nn.Linear(vec_dim, d_k)
        self.w_v = nn.Linear(vec_dim, d_k)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        z = torch.matmul(scores, v)
        return (z, scores)

    def forward(self, x):
        # Calculate query, key, and value matrices
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        z, scores = self.attention(q, k, v)

        return z, scores


if __name__ == '__main__':
    att = Attention(4)
    x = torch.Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    z = att(x)

    print(z)
    print(z.shape)