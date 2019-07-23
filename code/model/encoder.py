import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout, to_cuda=False):
        super(Encoder, self).__init__()
        self.fcs = []
        inp = input_size
        self.dropout = nn.Dropout(dropout)
        for hidden_layer in hidden_layers:
            self.fcs.append(nn.Linear(inp, hidden_layer))
            inp = hidden_layer
        self.output_fc = nn.Linear(inp, output_size)

        if to_cuda:
            for i in range(len(self.fcs)):
                self.fcs[i] = self.fcs[i].cuda()
            self.output_fc = self.output_fc.cuda()

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.output_fc(x)
        return x
