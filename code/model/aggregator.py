import torch
import torch.nn as nn
import torch.nn.functional as F


class AverageAggregator(nn.Module):
  def __init__(self, hidden_repr, to_cuda = False):
    super(AverageAggregator, self).__init__()

  def forward(self, x):
    x = torch.mean(x, 0)
    return x


class AttentionAggregator(nn.Module):
  def __init__(self, hidden_repr, to_cuda = False):
    super(AttentionAggregator, self).__init__()
    self.fc = nn.Linear(hidden_repr, hidden_repr)
    self.weight_vec = nn.Parameter(torch.FloatTensor(hidden_repr, 1))
    torch.nn.init.xavier_normal_(self.weight_vec)
    
    if to_cuda:
      self.fc = self.fc.cuda()

  def forward(self, x):
    energies = self.fc(x)
    energies = torch.matmul(energies, self.weight_vec)
    weights = F.softmax(energies, dim=0)

    x = x.permute([1, 0])  # transpose
    x = torch.matmul(x, weights)
    x = x.permute([1, 0])  # transpose

    return torch.sum(x, dim=0)