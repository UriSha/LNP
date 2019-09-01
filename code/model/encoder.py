import torch
import torch.nn as nn
import torch.nn.functional as F
from .aggregator import AverageAggregator
from .transformer import TransformerEncoder, TransformerEncoderLayer


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size, dropout, to_cuda=False):
        super(Encoder, self).__init__()
        self.fcs = nn.ModuleList()
        self.dps = nn.ModuleList()
        inp = input_size
        self.dropout = nn.Dropout(dropout)
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


    def forward(self, x, x_mask):
        for i in range(len(self.fcs)):
            x = self.fcs[i](x)
            x = F.relu(x)
            x = self.dps[i](x)
        x = self.output_fc(x)
        return x


class SelfAttentionEncoder(torch.nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward, dropout, num_layers, to_cuda=False):
        super(SelfAttentionEncoder, self).__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout), num_layers)

        if to_cuda:
            self.encoder = self.encoder.cuda()


    def forward(self, x, x_mask=None):
        return self.encoder(x, src_key_padding_mask=x_mask)


class LatentEncoder(torch.nn.Module):

    def __init__(self, d_model, nheads, num_hidden, num_latent, dim_feedforward, dropout, num_layers, to_cuda=False):
        super(LatentEncoder, self).__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout), num_layers)
        self.aggregator = AverageAggregator()
        self.mu = nn.Linear(num_hidden, num_latent)
        self.log_sigma = nn.Linear(num_hidden, num_latent)

        if to_cuda:
            self.encoder = self.encoder.cuda()
            self.aggregator = self.aggregator.cuda()
            self.mu = self.mu.cuda()
            self.log_sigma = self.log_sigma.cuda()


    def forward(self, x, x_mask=None):
        latent_representations = self.encoder(x, src_key_padding_mask=x_mask)
        latent_representations = latent_representations.transpose(0, 1)
        latent_aggregated_representation = self.aggregator(latent_representations, x_mask)

        # get mu and sigma
        mu = self.mu(latent_aggregated_representation)
        log_sigma = self.log_sigma(latent_aggregated_representation)

        # reparameterization trick
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)

        return z
