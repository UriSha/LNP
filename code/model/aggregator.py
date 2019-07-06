import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageAggregator(nn.Module):
    def __init__(self, hidden_repr, max_sent_len, to_cuda=False):
        super(AverageAggregator, self).__init__()
        self.max_sent_len = max_sent_len


    def forward(self, x, x_mask):
        r, c = x.shape
        x = x.view(r // self.max_sent_len, self.max_sent_len, c)

        weights = x_mask.float()
        weights = F.softmax(weights.masked_fill(x_mask, float('-inf')))
        weights = torch.unsqueeze(weights, dim=2)

        x = x.permute([0, 2, 1])  # transpose
        x = torch.matmul(x, weights)  # batch matrix multiplication
        x = x.permute([0, 2, 1])  # transpose
        
        return x


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_repr, max_sent_len, to_cuda=False):
        super(AttentionAggregator, self).__init__()
        self.max_sent_len = max_sent_len
        self.fc = nn.Linear(hidden_repr, hidden_repr)
        self.weight_vec = nn.Parameter(torch.FloatTensor(hidden_repr, 1))
        torch.nn.init.xavier_normal_(self.weight_vec)

        if to_cuda:
            self.fc = self.fc.cuda()


    def forward(self, x, x_mask):
        r, c = x.shape
        x = x.view(r // self.max_sent_len, self.max_sent_len, c)

        energies = self.fc(x)
        energies = torch.matmul(energies, self.weight_vec)
        energies.masked_fill_(x_mask, float('-inf'))
        weights = F.softmax(energies, dim=0)

        x = x.permute([0, 2, 1])  # transpose
        x = torch.matmul(x, weights)  # batch matrix multiplication
        x = x.permute([0, 2, 1])  # transpose

        return x