import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1, to_cuda=False):
        super().__init__()

        self.to_cuda = to_cuda

        self.embed_size = embed_size
        self.d_k = embed_size // num_heads
        self.heads = num_heads

        self.q_linear = nn.Linear(embed_size, embed_size)
        self.v_linear = nn.Linear(embed_size, embed_size)
        self.k_linear = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(embed_size, embed_size)

        if to_cuda:
            self.q_linear = self.q_linear.cuda()
            self.v_linear = self.v_linear.cuda()
            self.k_linear = self.k_linear.cuda()
            self.dropout = self.dropout.cuda()
            self.out = self.out.cuda()

    def forward(self, q, k, v, context_mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k)

        # transpose to get dimensions bs * h * sl * embed_size

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, context_mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.embed_size)

        output = self.out(concat)

        return output

    def attention(self, q, k, v, d_k, context_mask=None, dropout=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if context_mask is not None:
            head_count = q.shape[1]
            word_count = q.shape[2]
            adjusted_context_mask = self.create_adjusted_context_mask(context_mask, head_count, word_count)

            masked_scores = scores.masked_fill(adjusted_context_mask == 1, -1e9)

            soft_max_scores = F.softmax(masked_scores, dim=-1)
            # oposite_mask = (1 - adjusted_context_mask).clone().detach().float()
            # soft_max_scores = torch.min(soft_max_scores, oposite_mask)
        else:
            soft_max_scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            soft_max_scores = dropout(soft_max_scores)

        res = torch.matmul(soft_max_scores, v)
        return res

    def create_adjusted_context_mask(self, context_mask, head_count, word_count):
        # mask_first_squeeze = context_mask.unsqueeze(1)
        # mask_second_squeeze = mask_first_squeeze.unsqueeze(2)

        # mask_first_duplicate = mask_second_squeeze.repeat_interleave(self.heads, dim=1)
        # mask_second_duplicate = mask_first_duplicate.repeat_interleave(mask_first_duplicate.size(-1), dim=2)

        # # adjusted_context_mask = MultiHeadAttention.tensor_tile(adjusted_context_mask, 1, self.heads, self.to_cuda)
        # # adjusted_context_mask = MultiHeadAttention.tensor_tile(adjusted_context_mask, 2, adjusted_context_mask.size(-1), self.to_cuda)

        # adjusted_context_mask = torch.max(mask_second_duplicate, mask_second_duplicate.transpose(-2,-1))

        adjusted_context_mask = context_mask.unsqueeze(dim=1)
        adjusted_context_mask = adjusted_context_mask.repeat(1, head_count, 1)
        adjusted_context_mask = adjusted_context_mask.unsqueeze(dim=2)
        adjusted_context_mask = adjusted_context_mask.repeat(1, 1, word_count, 1)
        return adjusted_context_mask

    @staticmethod
    def tensor_tile(input_tensor, dim, n_tile, to_cuda):
        init_dim = input_tensor.size(dim)
        repeat_idx = [1] * input_tensor.dim()
        repeat_idx[dim] = n_tile
        input_tensor = input_tensor.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        if to_cuda:
            order_index = order_index.cuda()
        return torch.index_select(input_tensor, dim, order_index)
