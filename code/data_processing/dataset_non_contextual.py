import math
import torch
import random
from torch.utils.data import Dataset


class DatasetNonContextual(Dataset):
    def __init__(self, sents, w2id, id2w, max_seq_len, mask_ratios, transform=None, to_cuda=True):
        self.sents = sents
        self.transform = transform
        self.mask_ratios = mask_ratios
        self.w2id = w2id
        self.id2w = id2w
        self.max_seq_len = max_seq_len
        self.max_masked_size = int(math.ceil(max_seq_len * max(mask_ratios)))
        self.to_cuda = to_cuda
        self.mem = {}
        self.current_mask_ratio_index = 0


    def __getitem__(self, index):
        if index in self.mem:
            return self.mem[index]

        sent = self.sents[index]
        self.mem[index] = self.mask_sent(sent)
        return self.mem[index]


    def __len__(self):
        return len(self.sents)


    def mask_sent(self, sent):
        context_xs = [self.max_seq_len] * self.max_seq_len
        context_ys = [0] * self.max_seq_len
        context_mask = [1] * self.max_seq_len
        target_xs = [self.max_seq_len] * self.max_masked_size
        target_ys = [0] * self.max_masked_size
        target_mask = [1] * self.max_masked_size

        mask_ratio = self.mask_ratios[self.current_mask_ratio_index]
        self.current_mask_ratio_index = (self.current_mask_ratio_index + 1) % len(self.mask_ratios)

        num_of_masks = int(len(sent) * mask_ratio)
        num_of_masks = max(1, num_of_masks)

        indices_to_mask = sorted(random.sample(range(len(sent)), num_of_masks))

        j = 0
        k = 0
        for i in range(len(sent)):
            if j < len(indices_to_mask) and i == indices_to_mask[j]:
                target_xs[j] = indices_to_mask[j]
                target_ys[j] = sent[indices_to_mask[j]]
                target_mask[j] = 0
                j += 1
            else:
                context_xs[k] = i
                context_ys[k] = sent[i]
                context_mask[k] = 0
                k += 1

        context_xs = torch.LongTensor(context_xs)
        context_ys = torch.LongTensor(context_ys)
        context_mask = torch.ByteTensor(context_mask)
        target_xs = torch.LongTensor(target_xs)
        target_ys = torch.LongTensor(target_ys)
        target_mask = torch.ByteTensor(target_mask)
        

        if self.to_cuda:
            context_xs = context_xs.cuda()
            context_ys = context_ys.cuda()
            target_xs = target_xs.cuda()
            target_ys = target_ys.cuda()


        return context_xs, context_ys, context_mask, target_xs, target_ys, target_mask