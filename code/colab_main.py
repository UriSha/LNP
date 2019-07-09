import sys
import torch
import pickle as cPickle
import logging
import argparse
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from data_processing.dataset import text_dataset
from data_processing.dataset_once_random import text_dataset_once_random
from model.cnp import CNP
from data_processing.text_processor import TextProcessor
from training import Trainer


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

    text_processor = TextProcessor("data/APRC/{}".format(args.data_file), sents_limit=args.sent_count)

    if args.dataset_random_every_time:
        train_dataset = text_dataset(text_processor.train_sents, to_cuda=args.to_cuda)
    else:
        train_dataset = text_dataset_once_random(text_processor.train_sents, to_cuda=args.to_cuda)

    eval_dataset = text_dataset_once_random(text_processor.eval_sents, to_cuda=args.to_cuda)
    model = CNP(context_size=769, 
                target_size=1, 
                hidden_repr=800, 
                enc_hidden_layers=[800, 800], 
                dec_hidden_layers=[850, 1000], 
                output_size=len(text_processor.id2w), 
                max_sent_len=text_processor.max_seq_len, 
                max_target_size=text_processor.max_masked_size,
                to_cuda=args.to_cuda)
    trainer = Trainer(model, train_dataset, eval_dataset, args.batch_size,
                      args.learning_rate, args.epochs, args.to_cuda)
    trainer.run()
