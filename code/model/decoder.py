import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout, to_cuda=False):
        super(Decoder, self).__init__()
        self.fcs = []
        self.dps = []
        inp = input_size
        for hidden_layer in hidden_layers:
            self.fcs.append(nn.Linear(inp, hidden_layer))
            self.dps.append(nn.Dropout(dropout))
            inp = hidden_layer
        self.output_fc = nn.Linear(inp, output_size)

        if to_cuda:
            for i in range(len(self.fcs)):
                self.fcs[i] = self.fcs[i].cuda()
                self.dps[i] = self.dps[i].cuda()
            self.output_fc = self.output_fc.cuda()

    def forward(self, x):
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x = F.relu(x)
            x = self.dps[i](x)
        x = self.output_fc(x)
        return x
