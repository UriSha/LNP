import torch
import torch.nn as nn
import torchvision
from model.encoder import Encoder
from model.decoder import Decoder
from model.aggregator import AttentionAggregator, AverageAggregator


class CNP(nn.Module):
    def __init__(self, vec_size, hidden_repr, enc_hidden_layers, dec_hidden_layers, output_size, max_target_size, w2id, id2w, emb_weight, padding_idx, attn=False, to_cuda=False):
        super(CNP, self).__init__()
        self.encoder = Encoder(vec_size+1, enc_hidden_layers, hidden_repr, to_cuda)
        if attn:
            self.aggregator = AttentionAggregator(hidden_repr, to_cuda)
        else:
            self.aggregator = AverageAggregator(hidden_repr, to_cuda)
        self.decoder = Decoder(hidden_repr, 1, dec_hidden_layers, output_size, to_cuda)

        self.max_target_size = max_target_size
        self.w2id = w2id
        self.id2w = id2w
        self.embedding = nn.Embedding.from_pretrained(emb_weight, padding_idx=padding_idx)
        
        if to_cuda:
            self.encoder = self.encoder.cuda()
            self.aggregator = self.aggregator.cuda()
            self.decoder = self.decoder.cuda()


    def forward(self, context_ids, context_pos, context_mask, target):
        context = self.embedding(context_ids)
        context = torch.cat((context, context_pos.unsqueeze(dim=2)), dim=2)
        encodings = self.encoder(context)
        representations = self.aggregator(encodings, context_mask)

        x = self.concat_repr_to_target(representations, target)
        predictions = self.decoder(x)
        return predictions


    def concat_repr_to_target(self, representations, target):
        x = torch.repeat_interleave(representations, self.max_target_size, dim=1)
        target = torch.unsqueeze(target, dim=2)
        x = torch.cat((x, target), dim=2)
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
