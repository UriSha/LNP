import pickle as cPickle
import torch
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from data_processing.dataset import text_dataset
from model.cnp import CNP
from training import Trainer


def read_data(path):
    with open(path, "r") as f:
        text = f.readlines()
    text = [sent.rstrip("\n").split(" ") for sent in text]
    return text


if __name__ == "__main__":
    sents = read_data("data/APRC/APRC_small_mock.txt")
    to_cuda = True
    n_epoches = 2000
    dataset = text_dataset(sents, to_cuda)
    model = CNP(769, 1, 800, [700], [700], len(dataset.id2w), dataset.max_seq_len, dataset.max_masked_size, to_cuda=to_cuda)
    trainer = Trainer(model, dataset, None, 2, 0.001, n_epoches, to_cuda)
    trainer.run()