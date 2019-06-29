import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pickle

max_seq_length = 256


class text_dataset(Dataset):
    def __init__(self, x_list, to_cuda=True):
        self.x_list = x_list

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        self.bert_pretrained.eval()
        self.to_cuda = to_cuda

        if self.to_cuda:
            self.bert_pretrained.to('cuda')

    def __getitem__(self, index, transform):
        # Tokenized input
        text = self.tokenizer.tokenize(self.x_list[index])
        text.insert(0, "[CLS]")
        text.append("[SEP]")

        if len(text) > max_seq_length:
            text = text[:max_seq_length]

        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(text)

        padding = [0] * (max_seq_length - len(indexed_tokens))

        indexed_tokens += padding

        assert len(indexed_tokens) == max_seq_length

        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor(indexed_tokens)
        segments_tensors = torch.tensor([segments_ids])

        if self.to_cuda:
            # If we have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensors = segments_tensors.to('cuda')

        with torch.no_grad():
            # Predict hidden states features for each layer
            encoded_layers, = self.bert_pretrained(tokens_tensor, segments_tensors)
            # We have a hidden states for each of the 12 layers in model bert-base-uncased

        if transform:
            encoded_layers = transform(encoded_layers)

        return encoded_layers  # list_of_labels[0]

    def __len__(self):
        return len(self.x_list[0])


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
