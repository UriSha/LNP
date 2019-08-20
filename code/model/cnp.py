import math

from sklearn.preprocessing import normalize

from model.aggregator import AttentionAggregator, AverageAggregator
from model.cross_attention_aggregator import CrossAttentionAggregator
from model.decoder import Decoder
from model.encoder import Encoder
from model.latent_encoder import LatentEncoder

from model.transformer import *


class CNP(nn.Module):
    def __init__(self, embedding_size, hidden_repr, enc_hidden_layers, dec_hidden_layers, max_target_size, w2id,
                 id2w, emb_weight, padding_idx, max_seq_len, use_weight_matrix, nheads=2, use_pos_embedding=True,
                 dropout=0.1, attn=False, concat_embeddings=False, normalize_weights=True, to_cuda=False):
        super(CNP, self).__init__()
        self.use_pos_embedding = use_pos_embedding
        self.attn = attn

        self.concat_embeddings = concat_embeddings

        if use_pos_embedding:
            pos_embedding_size = embedding_size
        else:
            pos_embedding_size = 1

        if use_weight_matrix:
            output_size = emb_weight.shape[1]
        else:
            output_size = emb_weight.shape[0] - 1

        if concat_embeddings:
            input_size = embedding_size + pos_embedding_size
        else:
            input_size = embedding_size

        if attn:
            # self.encoder = SelfAttentionEncoderLayer(input_size=input_size, heads=2, dropout=dropout, to_cuda=to_cuda)

            # self.encoder = Transformer(input_size, nhead=nheads, num_encoder_layers=len(enc_hidden_layers), num_decoder_layers=len(enc_hidden_layers), dim_feedforward=enc_hidden_layers[0], dropout=dropout)
            # self.aggregator = None
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(input_size, nhead=nheads, dim_feedforward=enc_hidden_layers[0],
                                        dropout=dropout), num_layers=len(enc_hidden_layers))
            self.aggregator = CrossAttentionAggregator(embedding_size, nheads, dropout, to_cuda)

            self.latent_encoder = LatentEncoder(d_model=input_size,
                                                nhead=nheads,
                                                num_hidden=len(enc_hidden_layers),
                                                num_latent=len(enc_hidden_layers),
                                                dim_feedforward=enc_hidden_layers[0],
                                                dropout=dropout)
            # self.latent_encoder = TransformerEncoder(
            #     TransformerEncoderLayer(input_size, nhead=nheads, dim_feedforward=enc_hidden_layers[0],
            #                             dropout=dropout), num_layers=len(enc_hidden_layers))
            # self.latent_aggregator = AverageAggregator()

            self.decoder = Decoder(input_size + embedding_size, dec_hidden_layers, output_size, dropout, to_cuda)
        else:
            self.encoder = Encoder(input_size, enc_hidden_layers, hidden_repr, dropout, to_cuda)
            self.aggregator = AttentionAggregator(hidden_repr, to_cuda)
            # self.aggregator = AverageAggregator()
            self.latent_encoder = None
            self.latent_aggregator = None
            self.decoder = Decoder(input_size, dec_hidden_layers, output_size, dropout, to_cuda)

        self.max_target_size = max_target_size
        self.max_seq_len = max_seq_len
        self.w2id = w2id
        self.id2w = id2w

        self.embedding = nn.Embedding.from_pretrained(emb_weight, padding_idx=padding_idx)

        self.embedding_matrix = None
        if use_weight_matrix:
            embedding_matrix = emb_weight[1:].permute([1, 0])  # skip padding
            # normalize matrix by columns. for rows change to: axis=1
            if normalize_weights:
                self.embedding_matrix = torch.FloatTensor(normalize(embedding_matrix, axis=0, norm='l2'))
            else:
                self.embedding_matrix = torch.FloatTensor(embedding_matrix)
            self.embedding_matrix.requires_grad = False

        pos_embeddings_matrix = self.create_pos_embeddings_matrix(
            max_seq_len, embedding_size)
        self.pos_embeddings = nn.Embedding.from_pretrained(
            pos_embeddings_matrix, padding_idx=max_seq_len)
        self.pos_embeddings.requires_grad = False

        if to_cuda:
            if self.embedding_matrix is not None:
                self.embedding_matrix = self.embedding_matrix.cuda()

            self.embedding = self.embedding.cuda()
            self.encoder = self.encoder.cuda()
            if self.aggregator is not None:
                self.aggregator = self.aggregator.cuda()

            if self.latent_encoder is not None:
                self.latent_encoder = self.latent_encoder.cuda()

            if self.latent_aggregator is not None:
                self.latent_aggregator = self.latent_aggregator.cuda()

            self.decoder = self.decoder.cuda()
            self.pos_embeddings = self.pos_embeddings.cuda()

    def forward(self, context_ids, context_pos, context_mask, target, target_mask):
        sent_embeddings = self.embedding(context_ids)
        if self.use_pos_embedding:
            pos_embeddings = self.pos_embeddings(context_pos)
        else:
            pos_embeddings = context_pos.unsqueeze(dim=2).float()

        if self.concat_embeddings:
            context = torch.cat((sent_embeddings, pos_embeddings), dim=2)
        else:
            context = sent_embeddings + pos_embeddings

        if self.use_pos_embedding:
            emb_target = self.pos_embeddings(target)
        else:
            emb_target = target.unsqueeze(dim=2).float()

        if self.attn:
            context = context.transpose(0, 1)
            encodings = self.encoder(context, src_key_padding_mask=context_mask)
            encodings = encodings.transpose(0, 1)
            representations = self.aggregator(q=emb_target, k=pos_embeddings, r=encodings, context_mask=context_mask,
                                              target_mask=target_mask)

            # latent path- new
            prior_mu, prior_var, prior = self.latent_encoder(context, context_mask)

            train = 1
            # For training
            if train is not None:       # TODO- check how to change it
                posterior_mu, posterior_var, posterior = self.latent_encoder(emb_target, target)
                z = posterior

            # For Generation
            else:
                z = prior

            # latent path- old
            # latent_encodings = self.latent_encoder(context, src_key_padding_mask=context_mask)
            # latent_encodings = latent_encodings.transpose(0, 1)
            # latent_representations = self.latent_aggregator(latent_encodings, context_mask)

            # representations = self.encoder(context.transpose(0, 1), emb_target.transpose(0, 1), src_key_padding_mask=context_mask, tgt_key_padding_mask=target_mask)
            # representations = representations.transpose(0, 1)

            if self.concat_embeddings:
                x = torch.cat((representations, emb_target), dim=2)
            else:
                x = representations + emb_target

            latent_representations = torch.repeat_interleave(z, self.max_target_size, dim=1)
            x = torch.cat((x, latent_representations), dim=2)
        else:
            encodings = self.encoder(context, context_mask)
            representations = self.aggregator(encodings, context_mask)
            x = self.repeat_and_merge(representations, emb_target)

        predicted_embeddings = self.decoder(x)

        if self.embedding_matrix is not None:
            predicted_embeddings = torch.matmul(predicted_embeddings, self.embedding_matrix)

        return predicted_embeddings

    def create_pos_embeddings_matrix(self, max_seq_len, embed_size):
        pe = torch.zeros(max_seq_len + 1, embed_size)
        for pos in range(max_seq_len):
            for i in range(0, embed_size, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        return pe

    def repeat_and_merge(self, representations, target):
        x = torch.repeat_interleave(representations, self.max_target_size, dim=1)
        # target = torch.unsqueeze(target, dim=2)
        if self.concat_embeddings:
            x = torch.cat((x, target), dim=2)
        else:
            x = representations + target
        return x

    def eval_model(self):
        self.eval()
        self.encoder.eval()
        self.decoder.eval()
        if self.aggregator:
            self.aggregator.eval()

    def train_model(self):
        self.train()
        self.encoder.train()
        self.decoder.train()
        if self.aggregator:
            self.aggregator.train()
