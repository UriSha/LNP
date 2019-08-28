import torch.nn as nn


class CrossAttentionAggregator(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1, to_cuda=False):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, dropout=dropout)
        if to_cuda:
            self.multihead_attention = self.multihead_attention.cuda()

    def forward(self, q, k, r, context_mask, target_mask):
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        r = r.transpose(0, 1)
        rep = self.multihead_attention(q, k, r, context_mask)
        rep = rep[0].transpose(0, 1)

        return rep
