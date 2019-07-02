import torch
import torch.nn as nn
import torchvision
from encoder import Encoder
from decoder import Decoder
from aggregator import AttentionAggregator, AverageAggregator


class CNP(nn.Module):
    def __init__(self, context_size, target_size, hidden_repr, enc_hidden_layers, dec_hidden_layers, output_size, attn=False, to_cuda=False):
        super(CNP, self).__init__()
        self.encoder = Encoder(
            context_size, enc_hidden_layers, hidden_repr, to_cuda)
        if attn:
            self.aggregator = AttentionAggregator(hidden_repr, to_cuda)
        else:
            self.aggregator = AverageAggregator(hidden_repr, to_cuda)
        self.decoder = Decoder(hidden_repr, target_size,
                               dec_hidden_layers, output_size, to_cuda)
        
        if to_cuda:
            self.encoder = self.encoder.cuda()
            self.aggregator = self.aggregator.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, context, target):
        encodings = self.encoder(context)
        representation = self.aggregator(encodings)

        x = self.concat_repr_to_target(representation, target)
        predictions = self.decoder(x)
        return predictions

    def concat_repr_to_target(self, representation, target):
        x = representation.repeat(target.shape[0], 1)
        x = torch.cat((x, target), dim=1)
        return x


    def eval_model(self):
        self.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.aggregator.eval()

    def train_model(self):
        self.train()
        self.encoder.train()
        self.decoder.train()
        self.aggregator.train()
