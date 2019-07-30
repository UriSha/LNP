import torch.nn as nn
import torch.nn.functional as F

from model.multi_headed_attention import MultiHeadAttention
from model.norm import Norm


# build an encoder layer with one multi-head attention layer and one feed-forward layer
class SelfAttentionEncoderLayer(nn.Module):
    def __init__(self, context_size, target_size, heads, dropout, to_cuda):
        super().__init__()
        # embed_size = context_size + target_size
        embed_size = context_size
        self.norm_1 = Norm(embed_size)
        self.norm_2 = Norm(embed_size)

        self.self_attn = MultiHeadAttention(embed_size=embed_size, num_heads=heads, to_cuda=to_cuda)
        self.ff = FeedForward(embed_size=embed_size, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        if to_cuda:
            self.norm_1 = self.norm_1.cuda()
            self.norm_2 = self.norm_2.cuda()
            self.self_attn = self.self_attn.cuda()
            self.ff = self.ff.cuda()
            self.dropout_1 = self.dropout_1.cuda()
            self.dropout_2 = self.dropout_2.cuda()

    def forward(self, x, context_mask):
        x1 = self.norm_1(x)

        z = self.dropout_1(self.self_attn(q=x1, k=x1, v=x1, context_mask=context_mask))

        x2 = self.norm_2(x1 + z)

        x3 = x2 + self.dropout_2(self.ff(x2))

        return x3


class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout, d_ff=1024):
        super().__init__()
        self.linear_1 = nn.Linear(embed_size, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, embed_size)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
