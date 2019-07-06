import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle
import math
import random


# max_seq_length = 256

class text_dataset(Dataset):
    def __init__(self, text_as_list, mask_ratio=.25, transform=None, to_cuda=True):
        self.text = text_as_list

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        self.bert_pretrained.eval()
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.w2id, self.id2w, self.max_seq_len = self.initiate_vocab()
        self.max_masked_size = math.ceil(self.max_seq_len * mask_ratio)
        self.to_cuda = to_cuda

        if self.to_cuda:
            self.bert_pretrained.to('cuda')

    def __getitem__(self, index):
        # Tokenized input
        sentence = self.text[index].copy()
        print("sent", sentence)

        sent, masked_indices, target_xs, target_ys = self.mask_sent(sentence)
        sent.insert(0, "[CLS]")
        sent.append("[SEP]")

        sent = " ".join(sent)
        tokenized_sent = self.tokenizer.tokenize(sent)

        print("tokenized_sent", tokenized_sent)

        if len(tokenized_sent) - 2 > self.max_seq_len:
            # should not get here
            print("Error: len(tokenized_sent) > self.max_seq_len")
            tokenized_sent = tokenized_sent[:self.max_seq_len]

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)
        #   print("indexed_tokens", indexed_tokens)

        #         pad_size =  (self.max_seq_len - len(indexed_tokens))
        #         padding = [0] * pad_size
        #         indexed_tokens += padding

        #  assert len(indexed_tokens) == self.max_seq_len

        segments_ids = [0] * len(indexed_tokens)
        #   print("segments_ids", segments_ids)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        if self.to_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')

        with torch.no_grad():
            encoded_layers, _ = self.bert_pretrained(tokens_tensor, segments_tensors)

        embbedings_per_token = self.mean_transform(encoded_layers)
        #   print("embbedings_per_token.shape:" ,embbedings_per_token.shape)
        #   print()

        masked_indices_set = set(masked_indices)
        anti_mask_indices = [i for i in range(1, embbedings_per_token.shape[1] - 1) if i not in masked_indices_set]

        embbedings_per_token_without_masked = self.remove_masked_embeddings(embbedings_per_token, anti_mask_indices)

        embbedings_per_token_without_masked = embbedings_per_token_without_masked.squeeze(0)

        embbedings_per_token_without_masked_paddded, paddings_mask, num_of_paddings = self.pad_embedded_sentence(
            embbedings_per_token_without_masked)
        anti_mask_indices += [-1] * num_of_paddings

        embbedings_per_token_without_masked_paddded = self.concatenate_original_indecies(
            embbedings_per_token_without_masked_paddded, anti_mask_indices)

        return embbedings_per_token_without_masked_paddded, paddings_mask, target_xs, target_ys

    def __len__(self):
        return len(self.text)

    def pad_embedded_sentence(self, embedded_sent):
        #  print("embedded_sent.shape:", embedded_sent.shape)

        num_of_paddings = self.max_seq_len - embedded_sent.shape[0]
        paddings = torch.zeros((num_of_paddings, embedded_sent.shape[1]))
        if self.to_cuda:
            paddings = paddings.cuda()
        padded_sent = torch.cat((embedded_sent, paddings), 0)
        paddings_mask = [0] * embedded_sent.shape[0] + [1] * num_of_paddings
        paddings_mask = torch.ByteTensor(paddings_mask)

        #         print("padded_sent.shape:", padded_sent.shape)
        #         print("padded_sent:", padded_sent)
        #         print()

        #         print("paddings_mask.shape:", paddings_mask.shape)
        #         print("paddings_mask:", paddings_mask)
        #         print()

        if self.to_cuda:
            #    padded_sent = padded_sent.cuda()
            paddings_mask = paddings_mask.cuda()
        return padded_sent, paddings_mask, num_of_paddings

    def initiate_vocab(self):
        max_len = 0
        words_count = 0
        sent_count = 0
        w2id = {}
        id2w = {}

        for sent in self.text:
            for w in sent:
                if w not in w2id:
                    w2id[w] = words_count
                    id2w[words_count] = w
                    words_count += 1

            sent_as_string = (" ").join(sent)
            tokenized_sent = self.tokenizer.tokenize(sent_as_string)
            max_len = max(max_len, len(tokenized_sent))

        # return w2id, id2w, max_len + 2 # plus 2 for 'CLS' and 'SEP'
        return w2id, id2w, max_len

    def mask_sent(self, sent):

        target_xs = []
        target_ys = []

        num_of_masks = math.floor(len(sent) * self.mask_ratio)

        indices_to_mask = sorted(random.sample(range(len(sent)), num_of_masks))

        for idx in indices_to_mask:
            target_xs.append(idx)
            target_ys.append(self.w2id[sent[idx]])
            sent[idx] = '[MASK]'

        target_padding = [-1 for _ in range(self.max_seq_len - len(target))]
        target_xs.extend(target_padding)
        target_ys.extend(target_padding)

        target_xs = torch.Tensor(target_xs)
        target_ys = torch.Tensor(target_ys)

        if self.to_cuda:
            target_xs = target_xs.cuda()
            target_ys = target_ys.cuda()

        return sent, indices_to_mask, target_xs, target_ys

    def remove_masked_embeddings(self, sent_embedings, indices):
        return sent_embedings[:, indices]

    def mean_transform(self, list_of_tensors):
        big_tensor = torch.stack(list_of_tensors)
        return big_tensor.mean(dim=0)

    def concatenate_original_indecies(self, embbedings_per_token_without_masked, indices):
        indices = torch.Tensor(indices).unsqueeze(1)
        if self.to_cuda:
            indices = indices.cuda()
        # print()
        #       print("indices.shape:", indices.shape)
        #       print("embbedings_per_token_without_masked.shape:", embbedings_per_token_without_masked.shape)

        return torch.cat((embbedings_per_token_without_masked, indices), 1)


# class ToTensor(object):
#     def __call__(self, sample):
#         return {'indexed_tokens': torch.tensor(sample["indexed_tokens"]),
#                 'segments_ids': torch.tensor(sample["segments_ids"]}
#
#
# class ToCuda(object):
#     def __call__(self, sample):
#         return {'image': sample['image'].cuda(),
#                 'label': sample['label'].cuda()}


# tans = transforms.Compose([
#
#     ToTensor(),
#     ToCuda()
#
# ])


# def mean_transform(list_of_tensors):
#     big_tensor = torch.stack(list_of_tensors)
#     return big_tensor.mean(dim=0)
#

if __name__ == "__main__":
    # with open("../../data/APRC/APRC.txt") as f:
    # text_as_list = [line.replace('\n','') for line in f.readlines()]
    # text = [[w for w in line.split(" ")] for line in text_as_list]
    # pickle.dump(text, open('../../data/APRC/APRC.pkl', 'wb'))

    text = pickle.load(open("../../data/APRC/APRC.pkl", 'rb'))
    dataset = text_dataset(text)

    sent = dataset[0]

    x = 4
