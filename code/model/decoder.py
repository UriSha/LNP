import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, hidden_repr, input_size, hidden_layers, output_size, to_cuda=False):
        super(Decoder, self).__init__()
        self.fcs = []
        inp = hidden_repr + input_size
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
        x = self.output_fc(x)
        return x
