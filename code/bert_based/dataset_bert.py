import math
import random

import torch
from torch.utils.data import Dataset


class DatasetBert(Dataset):
    def __init__(self, sents, max_seq_len, mask_ratios, id2w, tokenizer, random_every_time=False, to_cuda=True):
        self.sents = [["[CLS]"] + [id2w[word_id] for word_id in sent] + ["[SEP]"] for sent in sents]
        self.mask_ratios = mask_ratios
        self.max_seq_len = max_seq_len
        self.max_masked_size = int(math.ceil(max_seq_len * max(mask_ratios)))
        self.to_cuda = to_cuda
        self.mem = {}
        self.current_mask_ratio_index = 0
        self.random_every_time = random_every_time
        self.id2w = id2w
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if index in self.mem:
            return self.mem[index]

        sent = self.sents[index]
        tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor = self.mask_sent(sent)
        if not self.random_every_time:
            self.mem[index] = tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor
        return tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor

    def __len__(self):
        return len(self.sents)

    def mask_sent(self, sent):

        mask_ratio = self.mask_ratios[self.current_mask_ratio_index]
        self.current_mask_ratio_index = (self.current_mask_ratio_index + 1) % len(self.mask_ratios)

        sent_copy = " ".join(sent)
        tokenized_text = self.tokenizer.tokenize(sent_copy)

        original_sent_len = len(tokenized_text) - 2
        num_of_masks = int(original_sent_len * mask_ratio)
        num_of_masks = max(1, num_of_masks)

        indices_to_mask = sorted(random.sample(range(1, len(tokenized_text) - 1), num_of_masks))

        orig_masked_tokens_ids = []
        positions_to_predict = []
        for i in indices_to_mask:
            orig_masked_tokens_ids.append(tokenized_text[i])
            positions_to_predict.append(i)
            tokenized_text[i] = "[MASK]"

        indexed_masked_tokes = self.tokenizer.convert_tokens_to_ids(orig_masked_tokens_ids)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        indexed_masked_tokes_tensor = torch.LongTensor(indexed_masked_tokes)
        positions_to_predict_tensor = torch.LongTensor(positions_to_predict)


        if self.to_cuda:
            tokens_tensor = tokens_tensor.cuda()
            segments_tensors = segments_tensors.cuda()
            indexed_masked_tokes_tensor = indexed_masked_tokes_tensor.cuda()
            positions_to_predict_tensor = positions_to_predict_tensor.cuda()


        return tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor
