import math
import torch
import random
from torch.utils.data import Dataset

class DatasetNonContextual(Dataset):
    def __init__(self, text_as_list, w2id, id2w, max_seq_len, mask_ratios, transform=None,
                 to_cuda=True):
        self.data = text_as_list
        self.transform = transform
        self.mask_ratios = mask_ratios
        self.w2id = w2id
        self.id2w = id2w
        self.max_seq_len = max_seq_len
        self.max_masked_size = int(math.ceil(max_seq_len * max(mask_ratios)))
        self.to_cuda = to_cuda
        self.MASK_SYMBOL = '<MASK>'
        self.mem = {}
        self.current_mask_ratio_index = 0


    def __getitem__(self, index):
        if index in self.mem:
            return self.mem[index]

        sent = self.data[index].copy()

        # print("sent: ", sent)
        sent, masked_indices, target_xs, target_ys, target_xs_mask = self.mask_sent(sent)
        # print("masked_sent: ", sent)
        masked_indices_set = set(masked_indices)
        anti_mask_indices = [i for i in range(len(sent)) if i not in masked_indices_set]

        sent = [word for word in sent if word != self.MASK_SYMBOL]

        sent_ids = [self.w2id[word] for word in sent]

        sent_ids_tensor = torch.tensor(sent_ids)

        padded_sent_ids_tensor, paddings_mask, num_of_paddings = self.pad_embedded_sentence(
            sent_ids_tensor)

        padding_idx = self.max_seq_len
        anti_mask_indices += [padding_idx] * num_of_paddings
        anti_mask_indices = torch.LongTensor(anti_mask_indices)

        self.mem[index] = padded_sent_ids_tensor, anti_mask_indices, paddings_mask, target_xs, target_xs_mask, target_ys
        return self.mem[index]


    def __len__(self):
        return len(self.data)


    def pad_embedded_sentence(self, embedded_sent):

        num_of_paddings = self.max_seq_len - embedded_sent.shape[0]
        paddings = torch.zeros(num_of_paddings, dtype=torch.long)
        if self.to_cuda:
            paddings = paddings.cuda()
            embedded_sent = embedded_sent.cuda()
        padded_sent = torch.cat((embedded_sent, paddings), 0)
        paddings_mask = [0] * embedded_sent.shape[0] + [1] * num_of_paddings
        paddings_mask = torch.ByteTensor(paddings_mask)

        if self.to_cuda:
            paddings_mask = paddings_mask.cuda()
        return padded_sent, paddings_mask, num_of_paddings


    def mask_sent(self, sent):

        target_xs = []
        target_ys = []

        mask_ratio = self.mask_ratios[self.current_mask_ratio_index]
        num_of_masks = int(math.floor(len(sent) * mask_ratio))
        num_of_masks = max(1, num_of_masks)

        indices_to_mask = sorted(random.sample(range(len(sent)), num_of_masks))

        for idx in indices_to_mask:
            target_xs.append(idx)
            word = sent[idx]
            word_id = self.w2id[word]
            target_ys.append(word_id)
            sent[idx] = self.MASK_SYMBOL

        target_xs_padding = [self.max_seq_len for _ in range(self.max_masked_size - len(target_xs))]
        target_ys_padding = [0 for _ in range(self.max_masked_size - len(target_xs))]
        target_xs.extend(target_xs_padding)
        target_ys.extend(target_ys_padding)

        target_xs_mask = [1 if p == self.max_seq_len else 0 for p in target_xs]
        target_xs = torch.LongTensor(target_xs)
        target_xs_mask = torch.ByteTensor(target_xs_mask)
        target_ys = torch.tensor(target_ys)

        if self.to_cuda:
            target_xs = target_xs.cuda()
            target_ys = target_ys.cuda()

        self.current_mask_ratio_index = (self.current_mask_ratio_index + 1) % len(self.mask_ratios)

        return sent, indices_to_mask, target_xs, target_ys, target_xs_mask