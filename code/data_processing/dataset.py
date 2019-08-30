import math
import torch
import random
from torch.utils.data import Dataset


class DatasetNonContextual(Dataset):
    
    def __init__(self, sents, max_seq_len, mask_ratios, random_every_time=False, to_cuda=True):
        self.sents = sents
        self.mask_ratios = mask_ratios
        self.max_seq_len = max_seq_len
        self.max_masked_size = int(math.ceil(max_seq_len * max(mask_ratios)))
        self.to_cuda = to_cuda
        self.mem = {}
        self.current_mask_ratio_index = 0
        self.random_every_time = random_every_time


    def __getitem__(self, index):
        if index in self.mem:
            return self.mem[index]

        sent = self.sents[index]
        context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask = self.mask_sent(sent)
        if not self.random_every_time:
            self.mem[index] = context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask
        return context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask


    def __len__(self):
        return len(self.sents)


    def mask_sent(self, sent):
        context_x = [self.max_seq_len] * self.max_seq_len
        context_y = [0] * self.max_seq_len
        context_mask = [1] * self.max_seq_len
        target_x = [self.max_seq_len] * self.max_masked_size
        target_y = [0] * self.max_masked_size
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
                target_x[j] = indices_to_mask[j]
                target_y[j] = sent[indices_to_mask[j]]
                target_mask[j] = 0
                j += 1
            else:
                context_x[k] = i
                context_y[k] = sent[i]
                context_mask[k] = 0
                k += 1

        context_x = torch.LongTensor(context_x)
        context_y = torch.LongTensor(context_y)
        context_mask = torch.ByteTensor(context_mask)
        target_x = torch.LongTensor(target_x)
        target_y = torch.LongTensor(target_y)
        target_mask = torch.ByteTensor(target_mask)
        sent_x = torch.LongTensor([i if i < len(sent) else self.max_seq_len for i in range(self.max_seq_len)])
        sent_y = torch.LongTensor([sent[i] if i < len(sent) else 0 for i in range(self.max_seq_len)])
        sent_mask = torch.ByteTensor([0 if i < len(sent) else 1 for i in range(self.max_seq_len)])

        if self.to_cuda:
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()
            sent_x = sent_x.cuda()
            sent_y = sent_y.cuda()
            sent_mask = sent_mask.cuda()

        return context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask
