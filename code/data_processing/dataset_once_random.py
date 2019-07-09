import math

import pickle
import random
import torch
from pytorch_pretrained_bert import BertModel

from code.data_processing.abstract_dataset import Abstract_Dataset


# max_seq_length = 256

class text_dataset_once_random(Abstract_Dataset):

    def __init__(self, text_as_list, mask_ratio=.25, transform=None, to_cuda=True):
        super(text_dataset_once_random, self).__init__(text_as_list=text_as_list,
                                                       mask_ratio=mask_ratio,
                                                       transform=transform,
                                                       to_cuda=to_cuda)
        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = self.get_tokenizer()

        self.data = self.create_data(text_as_list, tokenizer)

    def create_data(self, text_as_list, tokenizer):
        bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        bert_pretrained.eval()

        if self.to_cuda:
            bert_pretrained.to('cuda')

        data = []
        for original_sent in text_as_list:
            data_instance = self.generate_data_instance_fron_sentence(original_sent=original_sent,
                                                                      tokenizer=tokenizer,
                                                                      bert_pretrained=bert_pretrained)
            data.append(data_instance)

        return data

    def __getitem__(self, index):
        return self.data[index]




