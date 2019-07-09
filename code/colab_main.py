import sys
import torch
import pickle as cPickle
import logging
import argparse
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from data_processing.dataset import text_dataset
from data_processing.dataset_once_random import text_dataset_once_random
from sklearn.model_selection import train_test_split
from model.cnp import CNP
from training import Trainer


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
    parser.add_argument('-sc', '--sent_count',
                        help="sent_count (default: no limit)",
                        default=0,
                        type=int)
    parser.add_argument('-bs', '--batch_size',
                        help="batch_size (default: 16)",
                        default=16,
                        type=int)
    args = parser.parse_args()

    sents = read_data("data/APRC/{}.txt".format(args.data_file))

    if args.sent_count == 0:
        sent_count = len(sents)
    else:
        sent_count = args.sent_count
    train_sents, eval_sents = train_test_split(
        sents[:sent_count], test_size=0.1)
    if args.dataset_random_every_time:
        train_dataset = text_dataset(train_sents, to_cuda=args.to_cuda)
    else:
        train_dataset = text_dataset_once_random(
            train_sents, to_cuda=args.to_cuda)

    eval_dataset = text_dataset_once_random(eval_sents, to_cuda=args.to_cuda)
    model = CNP(context_size=769, 
                target_size=1, 
                hidden_repr=800, 
                enc_hidden_layers=[800, 800], 
                dec_hidden_layers=[850, 1000], 
                output_size=len(dataset.id2w), 
                max_sent_len=dataset.max_seq_len, 
                max_target_size=dataset.max_masked_size,
                to_cuda=args.to_cuda)
    trainer = Trainer(model, train_dataset, eval_dataset, args.batch_size,
                      args.learning_rate, args.epochs, args.to_cuda)
    trainer.run()
