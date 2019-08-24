import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageAggregator(nn.Module):
    # def __init__(self, hidden_repr, to_cuda=False):
    def __init__(self):
        super(AverageAggregator, self).__init__()

    def forward(self, x, x_mask):
        weights = x_mask.float()
        weights = F.softmax(weights.masked_fill(x_mask, float('-inf')), dim=1)
        weights = torch.unsqueeze(weights, dim=2)

        # x = x.permute([0, 2, 1])  # transpose
        x = x.transpose(1, 2)
        x = torch.matmul(x, weights)  # batch matrix multiplication
        x = x.transpose(1, 2)
        # x = x.permute([0, 2, 1])  # transpose

        return x


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_repr, to_cuda=False):
        super(AttentionAggregator, self).__init__()

        self.fc = nn.Linear(hidden_repr, hidden_repr)
        self.weight_vec = nn.Parameter(torch.FloatTensor(hidden_repr, 1))
        torch.nn.init.xavier_normal_(self.weight_vec)

        if to_cuda:
            self.fc = self.fc.cuda()

    def forward(self, x, x_mask):
        energies = self.fc(x)
        energies = torch.matmul(energies, self.weight_vec)
        energies = energies.squeeze(dim=2)
        energies.masked_fill_(x_mask, float('-inf'))
        weights = F.softmax(energies, dim=1)
        weights = torch.unsqueeze(weights, dim=2)

        # x = x.permute([0, 2, 1])  # transpose
        x = x.transpose(1, 2)
        x = torch.matmul(x, weights)  # batch matrix multiplication
        x = x.transpose(1, 2)
        # x = x.permute([0, 2, 1])  # transpose

        return x
