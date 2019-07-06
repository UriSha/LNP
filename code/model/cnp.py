import torch
import torch.nn as nn
import torchvision
from encoder import Encoder
from decoder import Decoder
from aggregator import AttentionAggregator, AverageAggregator


class CNP(nn.Module):
    def __init__(self, context_size, target_size, hidden_repr, enc_hidden_layers, dec_hidden_layers, output_size, max_sent_len, max_target_size, attn=False, to_cuda=False):
        super(CNP, self).__init__()
        self.encoder = Encoder(
            context_size, enc_hidden_layers, hidden_repr, to_cuda)
        if attn:
            self.aggregator = AttentionAggregator(hidden_repr, max_sent_len, to_cuda)
        else:
            self.aggregator = AverageAggregator(hidden_repr, max_sent_len, to_cuda)
        self.decoder = Decoder(hidden_repr, target_size, dec_hidden_layers, output_size, to_cuda)

        self.max_target_size = max_target_size
        
        if to_cuda:
            self.encoder = self.encoder.cuda()
            self.aggregator = self.aggregator.cuda()
            self.decoder = self.decoder.cuda()


    def forward(self, context, context_mask, target):
        encodings = self.encoder(context)
        representations = self.aggregator(encodings, context_mask)

        x = self.concat_repr_to_target(representations, target)
        predictions = self.decoder(x)
        return predictions


    def concat_repr_to_target(self, representations, target):
        x = representations.repeat(self.max_target_size, 1)
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
