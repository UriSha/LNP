import torch.nn as nn
from multi_headed_attention import MultiHeadAttention


class CrossAttentionAggregator(nn.Module):
    """The Attention Aggregator module."""

    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, k, q, r):
        multihead_attention = MultiHeadAttention(embed_size=self.embed_size,
                                                 num_heads=self.num_heads,
                                                 dropout=self.dropout)

        rep = multihead_attention(q, k, r, self._num_heads)

        return rep
