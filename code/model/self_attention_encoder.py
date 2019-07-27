import torch
import torch.nn as nn
import torch.nn.functional as F

from norm import Norm
from multi_headed_attention import MultiHeadAttention
from feed_forward import FeedForward


# build an encoder layer with one multi-head attention layer and one feed-forward layer
class SelfAttentionEncoder(nn.Module):
    def __init__(self, embed_size, heads, dropout, to_cuda):
        super().__init__()
        self.norm_1 = Norm(embed_size)
        self.norm_2 = Norm(embed_size)
        self.attn = MultiHeadAttention(embed_size=embed_size, heads=heads)
        self.ff = FeedForward(embed_size=embed_size, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        if to_cuda:
            self.norm_1 = self.norm_1.cuda()
            self.norm_2 = self.norm_2.cuda()
            self.attn = self.attn.cuda()
            self.ff = self.ff.cuda()
            self.dropout_1 = self.dropout_1.cuda()
            self.dropout_2 = self.dropout_2.cuda()

    def forward(self, x, context_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(q=x2, k=x2, v=x2, context_mask=context_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

