import math
import torch
import random
from torch.utils.data import Dataset



class DatasetBert(Dataset):

    def __init__(self, sents, max_seq_len, mask_ratios, id2w, tokenizer, random_every_time=False, to_cuda=True):
        self.sents = sents
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
        tokens_tensor, segments_tensors, context_mask, target_mask = self.mask_sent(sent)
        if not self.random_every_time:
            self.mem[index] = tokens_tensor, segments_tensors, context_mask, target_mask
        return tokens_tensor, segments_tensors, context_mask, target_mask


    def __len__(self):
        return len(self.sents)


    def mask_sent(self, sent):

        mask_ratio = self.mask_ratios[self.current_mask_ratio_index]
        self.current_mask_ratio_index = (self.current_mask_ratio_index + 1) % len(self.mask_ratios)

        num_of_masks = int(len(sent) * mask_ratio)
        num_of_masks = max(1, num_of_masks)

        indices_to_mask = sorted(random.sample(range(len(sent)), num_of_masks))

        masked_sent = ["<PAD>"] * self.max_seq_len
        ys = ["<PAD>"] * self.max_seq_len
        context_mask = [1] * self.max_seq_len
        target_mask = [1] * self.max_masked_size

        j = 0
        k = 0
        for i in range(len(sent) + 1):
            if i == len(sent):
                masked_sent.insert(i, "[SEP]")
                target_mask.insert(j, 0)
                j+=1
            elif j < len(indices_to_mask) and i == indices_to_mask[j]:
                masked_sent[i] = "[MASK]"
                ys[j] = self.id2w[sent[i]]
                target_mask[j] = 0
                j += 1
            else:
                masked_sent[i] = self.id2w[sent[i]]
                context_mask[k] = 0
                k += 1

        masked_sent.insert(0 , "[CLS]")


        tokenized_text = self.tokenizer.tokenize(masked_sent)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        context_mask = torch.ByteTensor(context_mask)
        target_mask = torch.ByteTensor(target_mask)

        if self.to_cuda:
            tokens_tensor = tokens_tensor.cuda()
            segments_tensors = segments_tensors.cuda()
            context_mask = context_mask.cuda()
            target_mask = target_mask.cuda()

        return tokens_tensor, segments_tensors, context_mask, target_mask
