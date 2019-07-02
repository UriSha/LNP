import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle
import math


# max_seq_length = 256


class text_dataset(Dataset):
    def __init__(self, text_as_list, mask_ratio=.25, transform=None, to_cuda=True):
        self.text = text_as_list

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        # self.bert_pretrained.eval()
        self.transform = transform
        self.mask_ratio = mask_ratio
        self.max_seq_len = self.calculate_max_seq_len()
        self.max_masked_size = math.ceil(self.max_seq_len * mask_ratio)
        self.to_cuda = to_cuda

        # if self.to_cuda:
        #     self.bert_pretrained.to('cuda')

    def __getitem__(self, index):
        # Tokenized input
        sent = self.text[index]
        tokenized_sent = self.tokenizer.tokenize(sent)
        tokenized_sent.insert(0, "[CLS]")
        tokenized_sent.append("[SEP]")

        if len(tokenized_sent) > self.max_seq_len:
            # should not get here
            print("Error: len(tokenized_sent) > self.max_seq_len")
            tokenized_sent = tokenized_sent[:self.max_seq_len]

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_sent)

        padding = [0] * (self.max_seq_len - len(indexed_tokens))

        indexed_tokens += padding

        assert len(indexed_tokens) == self.max_seq_len

        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor([segments_ids])

        if self.to_cuda:
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')

        # if self.transform:
        #     # If we have a GPU, put everything on cuda
        #     raw_sample = {'indexed_tokens': indexed_tokens, 'segments_ids': segments_ids}
        #     tokens_tensor, segments_ids = self.transform(raw_sample)


            # tokens_tensor = tokens_tensor.to('cuda')
            # segments_tensors = segments_tensors.to('cuda')

            # with torch.no_grad():
            # Predict hidden states features for each layer
            # encoded_layers, = self.bert_pretrained(tokens_tensor, segments_tensors)
            # We have a hidden states for each of the 12 layers in model bert-base-uncased

        # if self.transform:
        #     encoded_layers = self.transform(encoded_layers)

        return tokens_tensor, segments_tensors  # list_of_labels[0]

    def __len__(self):
        return len(self.text)

    def calculate_max_seq_len(self):
        max_len = 0

        for sent in self.text:
            sent = (" ").join(sent)
            tokenized_sent = self.tokenizer.tokenize(sent)
            max_len = max(max_len, len(tokenized_sent))

        return max_len

#
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


def mean_transform(list_of_tensors):
    big_tensor = torch.stack(list_of_tensors)

    return big_tensor.mean(dim=0)


if __name__ == "__main__":
    # with open("../../data/APRC/APRC.txt") as f:
    # text_as_list = [line.replace('\n','') for line in f.readlines()]
    # text = [[w for w in line.split(" ")] for line in text_as_list]
    # pickle.dump(text, open('../../data/APRC/APRC.pkl', 'wb'))

    text = pickle.load(open("../../data/APRC/APRC.pkl", 'rb'))
    dataset = text_dataset(text)

    sent = dataset[0]

    x = 4
