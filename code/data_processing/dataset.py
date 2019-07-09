import math

import pickle
import random
import torch
from pytorch_pretrained_bert import BertModel

from code.data_processing.abstract_dataset import Abstract_Dataset


# max_seq_length = 256


class text_dataset(Abstract_Dataset):

    def __init__(self, text_as_list, mask_ratio=.25, transform=None, to_cuda=True):
        super(text_dataset, self).__init__(text_as_list=text_as_list,
                                           mask_ratio=mask_ratio,
                                           transform=transform,
                                           to_cuda=to_cuda)

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = self.get_tokenizer()
        self.bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        self.bert_pretrained.eval()

        if self.to_cuda:
            self.bert_pretrained.to('cuda')

    def __getitem__(self, index):
        return self.generate_data_instance_fron_sentence(original_sent=self.data[index],
                                                         tokenizer=self.tokenizer,
                                                         bert_pretrained=self.bert_pretrained)

