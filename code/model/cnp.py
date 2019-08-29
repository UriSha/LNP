import math
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from model.encoder import Encoder
from model.decoder import Decoder
from model.aggregator import AttentionAggregator
from model.transformer import TransformerEncoder, TransformerEncoderLayer
from model.cross_attention_aggregator import CrossAttentionAggregator


class CNP(nn.Module):
    def __init__(self, hidden_repr, enc_hidden_layers, dec_hidden_layers, emb_weight,
                       max_seq_len, use_weight_matrix, nheads=2, use_pos_embedding=True, dropout=0.1,
                       attn=False, concat_embeddings=False, normalize_weights=True, to_cuda=False):
        super(CNP, self).__init__()

        self.attn = attn
        self.max_seq_len = max_seq_len
        self.concat_embeddings = concat_embeddings

        embedding_size = emb_weight.shape[1]
        pos_embedding_size = embedding_size if use_pos_embedding else 1
        output_size = embedding_size if use_weight_matrix else emb_weight.shape[0] - 1
        input_size = embedding_size + pos_embedding_size if concat_embeddings else embedding_size

        if attn:
            self.encoder = TransformerEncoder(TransformerEncoderLayer(input_size, nhead=nheads, dim_feedforward=enc_hidden_layers[0], dropout=dropout), num_layers=len(enc_hidden_layers))
            self.aggregator = CrossAttentionAggregator(embedding_size, nheads, dropout, to_cuda)
            self.decoder = Decoder(input_size, dec_hidden_layers, output_size, dropout, to_cuda)
        else:
            self.encoder = Encoder(input_size, enc_hidden_layers, hidden_repr, dropout, to_cuda)
            self.aggregator = AttentionAggregator(hidden_repr, to_cuda)
            self.decoder = Decoder(input_size, dec_hidden_layers, output_size, dropout, to_cuda)

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

        self.pos_embeddings = None
        if use_pos_embedding:
            pos_embedding_matrix = self.__create_pos_embeddings_matrix(max_seq_len, embedding_size)
            self.pos_embeddings = nn.Embedding.from_pretrained(pos_embedding_matrix, padding_idx=max_seq_len)
            self.pos_embeddings.requires_grad = False

        if to_cuda:
            self.word_embeddings = self.word_embeddings.cuda()
            self.encoder = self.encoder.cuda()
            self.aggregator = self.aggregator.cuda()
            self.decoder = self.decoder.cuda()
            if self.embedding_matrix is not None:
                self.embedding_matrix = self.embedding_matrix.cuda()
            if self.pos_embeddings is not None:
                self.pos_embeddings = self.pos_embeddings.cuda()


    def forward(self, context_xs, context_ys, context_mask, target_xs, target_mask):

        context_word_embeddings = self.word_embeddings(context_ys)
        context_pos_embeddings = self.pos_embeddings(context_xs) if self.pos_embeddings is not None else context_xs.unsqueeze(dim=2).float()

        if self.concat_embeddings:
            context = torch.cat((context_word_embeddings, context_pos_embeddings), dim=2)
        else:
            context = context_word_embeddings + context_pos_embeddings

        target_pos_embeddings = self.pos_embeddings(target_xs) if self.pos_embeddings is not None else target_xs.unsqueeze(dim=2).float()

        if self.attn:
            context = context.transpose(0, 1)
            context_encodings = self.encoder(context, src_key_padding_mask=context_mask)
            context_encodings = context_encodings.transpose(0, 1)
            representations = self.aggregator(q=target_pos_embeddings, k=context_pos_embeddings, r=context_encodings, context_mask=context_mask, target_mask=target_mask)

            if self.concat_embeddings:
                target = torch.cat((representations, target_pos_embeddings), dim=2)
            else:
                target = representations + target_pos_embeddings
        else:
            context_encodings = self.encoder(context, context_mask)
            representations = self.aggregator(context_encodings, context_mask)
            
            if self.concat_embeddings:
                representations = torch.repeat_interleave(representations, target_pos_embeddings.shape[1], dim=1)
                target = torch.cat((representations, target_pos_embeddings), dim=2)
            else:
                target = representations + target_pos_embeddings

        predictions = self.decoder(target)

        if self.embedding_matrix is not None:
            predictions = torch.matmul(predictions, self.embedding_matrix)

        return predictions


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
                    math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * i) / embed_size)))

        return pe
