import math

from sklearn.preprocessing import normalize

from model.aggregator import AttentionAggregator, AverageAggregator
from model.cross_attention_aggregator import CrossAttentionAggregator
from model.decoder import Decoder
from model.encoder import Encoder
from model.latent_encoder import LatentEncoder

from model.transformer import *


class CNP(nn.Module):
    def __init__(self, embedding_size, hidden_repr, enc_hidden_layers, dec_hidden_layers, w2id,
                 id2w, emb_weight, padding_idx, max_seq_len, use_weight_matrix, nheads=2, use_pos_embedding=True, dropout=0.1, attn=False, concat_embeddings=False, normalize_weights=True, to_cuda=False):
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
                                                num_hidden=input_size,
                                                num_latent=input_size,
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

            # if self.latent_encoder is not None:
            #     self.latent_encoder = self.latent_encoder.cuda()

            if self.latent_aggregator is not None:
                self.latent_aggregator = self.latent_aggregator.cuda()

            self.decoder = self.decoder.cuda()
            self.pos_embeddings = self.pos_embeddings.cuda()

    def forward(self, context_ids, context_pos, context_mask, target, target_mask, target_ys=None):
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
            target_pos_embeddings = self.pos_embeddings(target)
        else:
            target_pos_embeddings = target.unsqueeze(dim=2).float()

        if self.attn:
            context = context.transpose(0, 1)
            encodings = self.encoder(context, src_key_padding_mask=context_mask)
            encodings = encodings.transpose(0, 1)
            representations = self.aggregator(q=target_pos_embeddings, k=pos_embeddings, r=encodings, context_mask=context_mask,
                                              target_mask=target_mask)

            # latent path- new
            prior_mu, prior_var, prior = self.latent_encoder(context, context_mask)

            # For training
            if target_ys is not None:
                full_sentence = torch.zeros(context_mask.shape).long().cuda()

                for batch in range(context_mask.shape[0]):
                    for index in range(context_mask.shape[1]):
                        if context_mask[batch.cuda()][index.cuda()] == 1:
                            break
                        position_in_sent = context_pos[batch.cuda()][index.cuda()]
                        value = context_ids[batch.cuda()][index.cuda()]
                        full_sentence[batch.cuda()][position_in_sent.cuda()] = value.cuda()

                for batch in range(target_mask.shape[0]):
                    for index in range(target_mask.shape[1]):
                        if target_mask[batch.cuda()][index.cuda()] == 1:
                            break
                        position_in_sent = target[batch.cuda()][index.cuda()]
                        value = target_ys[batch.cuda()][index.cuda()]
                        full_sentence[batch.cuda()][position_in_sent.cuda()] = value

                sentence_positions = [i for i in range(full_sentence.shape[1])].cuda()
                batch_positions = [sentence_positions for _ in range(full_sentence.shape[0])]
                batch_positions = torch.LongTensor(batch_positions).cuda()

                sent_pos_embeddings = self.pos_embeddings(batch_positions).cuda()
                full_sentence = torch.LongTensor(full_sentence).cuda()

                target_word_embeddings = self.embedding(full_sentence).cuda()
                latent_target = sent_pos_embeddings + target_word_embeddings # position emb + word emb
                latent_target = latent_target.transpose(0, 1).cuda()
                # latent_target = torch.cat((latent_target, context), dim=0)
                # latent_mask = torch.cat((target_mask, context_mask), dim=1)
                posterior_mu, posterior_var, posterior = self.latent_encoder(latent_target)
                z = posterior

            # For Generation
            else:
                posterior_mu, posterior_var = None, None
                z = prior

            # latent path- old
            # latent_encodings = self.latent_encoder(context, src_key_padding_mask=context_mask)
            # latent_encodings = latent_encodings.transpose(0, 1)
            # latent_representations = self.latent_aggregator(latent_encodings, context_mask)

            # representations = self.encoder(context.transpose(0, 1), target_pos_embeddings.transpose(0, 1), src_key_padding_mask=context_mask, tgt_key_padding_mask=target_mask)
            # representations = representations.transpose(0, 1)

            if self.concat_embeddings:
                x = torch.cat((representations, target_pos_embeddings), dim=2)
            else:
                x = representations + target_pos_embeddings

            latent_representations = torch.repeat_interleave(z, self.max_target_size, dim=1)
            x = torch.cat((x, latent_representations), dim=2)
        else:
            encodings = self.encoder(context, context_mask)
            representations = self.aggregator(encodings, context_mask)
            x = self.repeat_and_merge(representations, target_pos_embeddings)

        predicted_embeddings = self.decoder(x)

        if self.embedding_matrix is not None:
            predicted_embeddings = torch.matmul(predicted_embeddings, self.embedding_matrix)

        return predicted_embeddings, prior_mu, prior_var, posterior_mu, posterior_var

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
        x = torch.repeat_interleave(representations, target.shape[1], dim=1)
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
