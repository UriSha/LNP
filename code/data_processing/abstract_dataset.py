import math

import random
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(self, text_as_list, tokenizer, w2id, max_seq_len, max_masked_size, mask_ratio=.25, transform=None,
                 to_cuda=True):
        print()
        print("init Dataset")
        self.data = text_as_list
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.w2id = w2id
        self.max_seq_len = max_seq_len
        self.max_masked_size = max_masked_size
        self.to_cuda = to_cuda

    def __getitem__(self, item):
        raise Exception('AbstractDataset has no __getitem__!')

    def generate_data_instance_fron_sentence(self, original_sent, tokenizer, bert_pretrained):
        sentence = original_sent.copy()

        sent, masked_indices, target_xs, target_ys = self.mask_sent(sentence)
        # print("masked sentance: ", sentence)
        sent.insert(0, "[CLS]")
        sent.append("[SEP]")

        sent = " ".join(sent)

        # Tokenized input
        tokenized_sent = tokenizer.tokenize(sent)

        if len(tokenized_sent) - 2 > self.max_seq_len:
            # should not get here
            print("Error: len(tokenized_sent) > self.max_seq_len")
            tokenized_sent = tokenized_sent[:self.max_seq_len]

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sent)

        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        if self.to_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = bert_pretrained(tokens_tensor, segments_tensors)

        embbedings_per_token = self.mean_transform(encoded_layers)

        masked_indices_set = set(masked_indices)
        anti_mask_indices = [i for i in range(1, embbedings_per_token.shape[1] - 1) if i not in masked_indices_set]

        embbedings_per_token_without_masked = self.remove_masked_embeddings(embbedings_per_token, anti_mask_indices)

        embbedings_per_token_without_masked = embbedings_per_token_without_masked.squeeze(0)

        embbedings_per_token_without_masked_paddded, paddings_mask, num_of_paddings = self.pad_embedded_sentence(
            embbedings_per_token_without_masked)
        anti_mask_indices += [-1] * num_of_paddings

        embbedings_per_token_without_masked_paddded = self.concatenate_original_indices(
            embbedings_per_token_without_masked_paddded, anti_mask_indices)

        return embbedings_per_token_without_masked_paddded, paddings_mask, target_xs, target_ys

    def __len__(self):
        return len(self.data)

    def get_tokenizer(self):
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def pad_embedded_sentence(self, embedded_sent):
        #  print("embedded_sent.shape:", embedded_sent.shape)

        num_of_paddings = self.max_seq_len - embedded_sent.shape[0]
        paddings = torch.zeros((num_of_paddings, embedded_sent.shape[1]))
        if self.to_cuda:
            paddings = paddings.cuda()
        padded_sent = torch.cat((embedded_sent, paddings), 0)
        paddings_mask = [0] * embedded_sent.shape[0] + [1] * num_of_paddings
        paddings_mask = torch.ByteTensor(paddings_mask)

        if self.to_cuda:
            paddings_mask = paddings_mask.cuda()
        return padded_sent, paddings_mask, num_of_paddings

    def mask_sent(self, sent):

        target_xs = []
        target_ys = []

        num_of_masks = int(math.floor(len(sent) * self.mask_ratio))
        num_of_masks = max(1, num_of_masks)

        indices_to_mask = sorted(random.sample(range(len(sent)), num_of_masks))

        for idx in indices_to_mask:
            target_xs.append(idx)
            target_ys.append(self.w2id[sent[idx]])
            sent[idx] = '[MASK]'

        target_padding = [-1 for _ in range(self.max_masked_size - len(target_xs))]
        target_xs.extend(target_padding)
        target_ys.extend(target_padding)

        target_xs = torch.FloatTensor(target_xs)
        target_ys = torch.tensor(target_ys)

        if self.to_cuda:
            target_xs = target_xs.cuda()
            target_ys = target_ys.cuda()

        return sent, indices_to_mask, target_xs, target_ys

    def remove_masked_embeddings(self, sent_embedings, indices):
        return sent_embedings[:, indices]

    def mean_transform(self, list_of_tensors):
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor.mean(dim=0)

    def concatenate_original_indices(self, embbedings_per_token_without_masked, indices):
        indices = torch.Tensor(indices).unsqueeze(1)
        if self.to_cuda:
            indices = indices.cuda()

        return torch.cat((embbedings_per_token_without_masked, indices), 1)

    def update_xs_tokenized_indices(self, target_xs, tokenized_sent):

        res = []
        for idx, token in enumerate(tokenized_sent):
            if token == '[MASK]':
                res.append(idx)

        for idx, x in enumerate(target_xs):
            if x == -1:
                res.extend(target_xs[idx:])
                break

        return res
