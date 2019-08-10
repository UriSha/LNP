import torch.nn as nn
from model.multi_headed_attention import MultiHeadAttention
from model.aggregator import *


class CrossAttentionAggregator(nn.Module):
    """The Attention Aggregator module."""

    def __init__(self, embed_size, num_heads, dropout=0.1, to_cuda=False):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.multihead_attention = MultiHeadAttention(embed_size=self.embed_size,
                                                 num_heads=self.num_heads,
                                                 dropout=self.dropout,
                                                 to_cuda=to_cuda)
        self.aggregator = AttentionAggregator(embed_size, to_cuda)
        # self.aggregator = AverageAggregator(embed_size, to_cuda)

    def forward(self, k, q, r, context_mask, target_mask):

        # rep = self.multihead_attention(q, k, r, context_mask)
        # rep = self.aggregator(rep, target_mask)
        rep = self.aggregator(r, context_mask)
        return rep
