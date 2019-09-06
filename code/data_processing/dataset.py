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
        # context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask = self.mask_sent(sent)

        src, src_mask, src_padding_mask, tgt, tgt_mask, tgt_padding_mask, sent_x, sent_y, sent_mask, target_x, target_y = self.get_src_and_tgt(sent=sent)

        if not self.random_every_time:
            # self.mem[index] = context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask
            self.mem[index] = src, src_mask, src_padding_mask, tgt, tgt_mask, tgt_padding_mask, sent_x, sent_y, sent_mask, target_x, target_y

        # return context_x, context_y, context_mask, target_x, target_y, target_mask, sent_x, sent_y, sent_mask
        return src, src_mask, src_padding_mask, tgt, tgt_mask, tgt_padding_mask, sent_x, sent_y, sent_mask, target_x, target_y


    def __len__(self):
        return len(self.sents)


    def get_src_and_tgt(self, sent):
        context_x = [self.max_seq_len] * self.max_seq_len
        context_y = [0] * self.max_seq_len
        context_mask = [1] * self.max_seq_len
        target_x = [self.max_seq_len] * self.max_masked_size
        target_y = [0] * self.max_masked_size
        target_mask = [1] * self.max_masked_size

        src = [0] * self.max_seq_len
        src_mask = [float('-inf')] * self.max_seq_len
        src_padding_mask = [1] * self.max_seq_len

        tgt = [0] * self.max_seq_len
        tgt_mask = [float('-inf')] * self.max_seq_len
        tgt_padding_mask = [1] * self.max_seq_len

        mask_ratio = self.mask_ratios[self.current_mask_ratio_index]
        self.current_mask_ratio_index = (self.current_mask_ratio_index + 1) % len(self.mask_ratios)

        num_of_masks = int(len(sent) * mask_ratio)
        num_of_masks = max(1, num_of_masks)

        indices_to_mask = sorted(random.sample(range(len(sent)), num_of_masks))

        j = 0
        for i in range(len(sent)):
            if j < len(indices_to_mask) and i == indices_to_mask[j]:
                src[i] = 0
                src_mask[i] = float('-inf')
                src_padding_mask[i] = 1
                j += 1
            else:
                src[i] = sent[i]
                src_mask[i] = 0

            tgt[i] = sent[i]
            tgt_mask[i] = 0

            tgt_padding_mask[i] = 0

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

        src = torch.LongTensor(src)
        src_mask = torch.Tensor(src_mask)
        src_padding_mask = torch.ByteTensor(src_padding_mask)

        tgt = torch.LongTensor(tgt)
        tgt_mask = torch.Tensor(tgt_mask)
        tgt_padding_mask = torch.ByteTensor(tgt_padding_mask)

        sent_x = torch.LongTensor([i if i < len(sent) else self.max_seq_len for i in range(self.max_seq_len)])
        sent_y = torch.LongTensor([sent[i] if i < len(sent) else 0 for i in range(self.max_seq_len)])
        sent_mask = torch.ByteTensor([0 if i < len(sent) else 1 for i in range(self.max_seq_len)])

        target_x = torch.LongTensor(target_x)
        target_y = torch.LongTensor(target_y)

        if self.to_cuda:
            src = src.cuda()
            src_mask = src_mask.cuda()
            src_padding_mask = src_padding_mask.cuda()

            tgt = tgt.cuda()
            tgt_mask = tgt_mask.cuda()
            tgt_padding_mask = tgt_padding_mask.cuda()

            sent_x = sent_x.cuda()
            sent_y = sent_y.cuda()
            sent_mask = sent_mask.cuda()

            target_x = torch.LongTensor(target_x)
            target_y = torch.LongTensor(target_y)

        return src, src_mask, src_padding_mask, tgt, tgt_mask, tgt_padding_mask, sent_x, sent_y, sent_mask, target_x, target_y


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


if __name__ == '__main__':
    # sents = [['I', 'love', 'this', 'book']]
    sents = [[19, 5, 7, 13]]
    max_seq_len = 5
    mask_ratios = [0.25]

    ds = DatasetNonContextual(sents, max_seq_len, mask_ratios, random_every_time=False, to_cuda=False)

    x = ds[0]


