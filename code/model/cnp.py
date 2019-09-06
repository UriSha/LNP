import math

import torch
import torch.nn as nn
from sklearn.preprocessing import normalize

from .transformer import Transformer


class CNP(nn.Module):

    def __init__(self, enc_hidden_layers, dec_hidden_layers, emb_weight,
                 max_seq_len, use_weight_matrix, use_latent, nheads=2, dropout=0.1,
                 normalize_weights=True, to_cuda=False):
        super(CNP, self).__init__()

        embedding_size = emb_weight.shape[1]
        output_size = embedding_size if use_weight_matrix else emb_weight.shape[0] - 1
        input_size = embedding_size

        # self.encoder = SelfAttentionEncoder(input_size, nheads, enc_hidden_layers[0], dropout, len(enc_hidden_layers), to_cuda)
        # self.aggregator = CrossAttentionAggregator(embedding_size, nheads, dropout, to_cuda)

        # if use_latent:
        #     self.latent_encoder = LatentEncoder(input_size, nheads, input_size, input_size, enc_hidden_layers[0], dropout, len(enc_hidden_layers), to_cuda)
        #     self.decoder = Decoder(2 * input_size, dec_hidden_layers, output_size, dropout, to_cuda)
        # else:
        #     self.latent_encoder = None
        #     self.decoder = Decoder(input_size, dec_hidden_layers, output_size, dropout, to_cuda)

        self.transformer = Transformer(input_size, nheads, len(enc_hidden_layers), len(dec_hidden_layers),
                                       enc_hidden_layers[0], dropout)

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
            # self.encoder = self.encoder.cuda()
            # self.aggregator = self.aggregator.cuda()
            # self.decoder = self.decoder.cuda()
            self.transformer = self.transformer.cuda()
            self.pos_embeddings = self.pos_embeddings.cuda()
            # if self.latent_encoder is not None:
            #     self.latent_encoder = self.latent_encoder.cuda()
            if self.embedding_matrix is not None:
                self.embedding_matrix = self.embedding_matrix.cuda()

    def forward(self, src, src_mask, src_padding_mask, tgt, tgt_mask, tgt_padding_mask, sent_x, sent_y, sent_mask):

        src = self.word_embeddings(src)
        tgt = self.word_embeddings(tgt)

        # pos_embeddings = self.pos_embeddings(sent_x)
        #
        # src = src_embeddings + pos_embeddings
        # tgt = tgt_embeddings + pos_embeddings

        src = src.transpose(0, 1)
        # context_encodings = self.encoder(context, context_mask)
        # context_encodings = context_encodings.transpose(0, 1)
        # representations = self.aggregator(q=target_pos_embeddings, k=context_pos_embeddings, r=context_encodings, context_mask=context_mask)

        tgt = tgt.transpose(0, 1)

        kl = None

        predictions = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        predictions = predictions.transpose(0, 1)

        if self.embedding_matrix is not None:
            predictions = torch.matmul(predictions, self.embedding_matrix)

        return predictions, kl

    def kl_div(self, prior_mu, prior_var, posterior_mu, posterior_var):
        kl_div = (torch.exp(posterior_var) + (posterior_mu - prior_mu) ** 2) / torch.exp(prior_var) - 1. + (
                    prior_var - posterior_var)
        kl_div = 0.5 * kl_div.sum()
        return kl_div

    def test_model(self):
        self.eval()
        # self.encoder.eval()
        # self.decoder.eval()
        # self.aggregator.eval()
        self.transformer.eval()

    def train_model(self):
        self.train()
        # self.encoder.train()
        # self.decoder.train()
        # self.aggregator.train()
        self.transformer.train()

    def __create_pos_embeddings_matrix(self, max_seq_len, embed_size):
        pe = torch.zeros(max_seq_len + 1, embed_size)
        for pos in range(max_seq_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** (i / embed_size)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** (i / embed_size)))

        return pe
