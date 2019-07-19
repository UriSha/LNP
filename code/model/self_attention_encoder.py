import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, to_cuda=False):
        super(Encoder, self).__init__()

        self.Wq = nn.Parameter(torch.FloatTensor(embed_size, hidden_size))
        nn.init.xavier_normal_(self.Wq)

        self.Wk = nn.Parameter(torch.FloatTensor(embed_size, hidden_size))
        nn.init.xavier_normal_(self.Wk)

        self.Wv = nn.Parameter(torch.FloatTensor(embed_size, hidden_size))
        nn.init.xavier_normal_(self.Wv)


    def forward(self, x):
        pass
