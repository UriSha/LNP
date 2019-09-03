import math

import torch
import torch.nn as nn
from sklearn.preprocessing import normalize

from .aggregator import CrossAttentionAggregator
from .decoder import Decoder
from .encoder import SelfAttentionEncoder, LatentEncoder


class CNP(nn.Module):

    def __init__(self, enc_hidden_layers, dec_hidden_layers, emb_weight,
                 max_seq_len, use_weight_matrix, use_latent, nheads=2, dropout=0.1,
                 normalize_weights=True, to_cuda=False):
        super(CNP, self).__init__()

        embedding_size = emb_weight.shape[1]
        output_size = embedding_size if use_weight_matrix else emb_weight.shape[0] - 1
        input_size = embedding_size

        self.encoder = SelfAttentionEncoder(input_size, nheads, enc_hidden_layers[0], dropout, len(enc_hidden_layers),
                                            to_cuda)
        self.aggregator = CrossAttentionAggregator(embedding_size, nheads, dropout, to_cuda)

        decoder_input_size = input_size
        if use_latent:
            decoder_input_size = 2 * input_size

        self.decoder = Decoder(decoder_input_size, dec_hidden_layers, output_size, dropout, to_cuda)
        if use_latent:
            self.latent_encoder = LatentEncoder(input_size, nheads, input_size, input_size, enc_hidden_layers[0],
                                                dropout, len(enc_hidden_layers), to_cuda)
        else:
            self.latent_encoder = None

        self.word_embeddings = nn.Embedding.from_pretrained(emb_weight, padding_idx=0)

        self.embedding_matrix = None
        if use_weight_matrix:
            embedding_matrix = emb_weight[1:].permute([1, 0])  # skip padding
            # normalize matrix by columns. for rows change to: axis=1
            if normalize_weights:
                self.embedding_matrix = torch.FloatTensor(normalize(embedding_matrix, axis=0, norm='l2'))
            else:
                self.embedding_matrix = torch.FloatTensor(embedding_matrix)
            self.embedding_matrix.requires_grad = False

        pos_embedding_matrix = self.__create_pos_embeddings_matrix(max_seq_len, embedding_size)
        self.pos_embeddings = nn.Embedding.from_pretrained(pos_embedding_matrix, padding_idx=max_seq_len)
        self.pos_embeddings.requires_grad = False

        if to_cuda:
            self.word_embeddings = self.word_embeddings.cuda()
            self.encoder = self.encoder.cuda()
            self.aggregator = self.aggregator.cuda()
            self.decoder = self.decoder.cuda()
            self.pos_embeddings = self.pos_embeddings.cuda()
            if self.latent_encoder is not None:
                self.latent_encoder = self.latent_encoder.cuda()
            if self.embedding_matrix is not None:
                self.embedding_matrix = self.embedding_matrix.cuda()

    def forward(self, context_xs, context_ys, context_mask, target_xs, target_mask, sents=None):

        context_word_embeddings = self.word_embeddings(context_ys)
        context_pos_embeddings = self.pos_embeddings(context_xs)
        context = context_word_embeddings + context_pos_embeddings
        target_pos_embeddings = self.pos_embeddings(target_xs)

        context = context.transpose(0, 1)
        context_encodings = self.encoder(context, context_mask)
        context_encodings = context_encodings.transpose(0, 1)
        representations = self.aggregator(q=target_pos_embeddings, k=context_pos_embeddings, r=context_encodings,
                                          context_mask=context_mask, target_mask=target_mask)

        target = representations + target_pos_embeddings

        if self.latent_encoder is not None:
            prior_mu, prior_var, prior = self.latent_encoder(context, context_mask)

            # For Training
            if sents is not None:
                sent_xs, sent_ys, sent_mask = sents[0], sents[1], sents[2]
                sent_pos_embeddings = self.pos_embeddings(sent_xs)
                sent_word_embeddings = self.word_embeddings(sent_ys)
                latent_target = sent_pos_embeddings + sent_word_embeddings
                latent_target = latent_target.transpose(0, 1)
                posterior_mu, posterior_var, posterior = self.latent_encoder(latent_target, sent_mask)
                z = posterior

            # For Generation
            else:
                z = prior

            latent_representations = torch.repeat_interleave(z, target_pos_embeddings.shape[1], dim=1)
            target = torch.cat((target, latent_representations), dim=2)

        predictions = self.decoder(target)

        if self.embedding_matrix is not None:
            predictions = torch.matmul(predictions, self.embedding_matrix)

        kl = None
        if self.latent_encoder is not None and sents is not None:
            kl = self.kl_div(prior_mu, prior_var, posterior_mu, posterior_var)

        return predictions, kl

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu-prior_mu) ** 2) / torch.exp(prior_var) - 1. + (prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

    def test_model(self):
        self.eval()
        self.encoder.eval()
        self.decoder.eval()
        self.aggregator.eval()

    def train_model(self):
        self.train()
        self.encoder.train()
        self.decoder.train()
        self.aggregator.train()

    def __create_pos_embeddings_matrix(self, max_seq_len, embed_size):
        pe = torch.zeros(max_seq_len + 1, embed_size)
        for pos in range(max_seq_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** (i / embed_size)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** (i / embed_size)))

        return pe

