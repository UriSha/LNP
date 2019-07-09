import pickle as cPickle
import torch
import logging
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from data_processing.dataset import text_dataset
from data_processing.dataset_once_random import text_dataset_once_random
from model.cnp import CNP
from training import Trainer
import sys
import argparse


def read_data(path):
    with open(path, "r") as f:
        text = f.readlines()
    text = [sent.rstrip("\n").split(" ") for sent in text]
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-da', '--data_file',
                        help="data_file (default: APRC_new1.txt)",
                        default='APRC_new1.txt')
    parser.add_argument('-e', '--epochs',
                        help="training epochs (default: 200)",
                        default=200,
                        type=int)
    parser.add_argument('-ds', '--dataset_random_every_time',
                        help="random mask every call for getitem or only at init (default: False)",
                        default=False,
                        type=bool)
    parser.add_argument('-lr', '--learning_rate',
                        help="learning rate (default: .0005)",
                        default=.0005,
                        type=float)
    parser.add_argument('-c', '--to_cuda',
                        help="to_cuda (default: True)",
                        default=True,
                        type=bool)
    args = parser.parse_args()

    sents = read_data("data/APRC/{}.txt".format(args.model))

    if args.dataset_random_every_time:
        dataset = text_dataset(sents, to_cuda=args.to_cuda)
    else:
        dataset = text_dataset_once_random(sents, to_cuda=args.to_cuda)

    model = CNP(769, 1, 800, [700], [700], len(dataset.id2w), dataset.max_seq_len, dataset.max_masked_size,
                to_cuda=args.to_cuda)
    trainer = Trainer(model, dataset, None, 2, args.learning_rate, args.epochs, args.to_cuda)
    trainer.run()
