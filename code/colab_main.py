import argparse

from data_processing.dataset_consistent import DatasetConsistent
from data_processing.dataset_random import DatasetRandom
from data_processing.text_processor import TextProcessor
from model.cnp import CNP
from training import Trainer


def parse_arguments():
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
    return parser.parse_args()


def main():
    args = parse_arguments()

    text_processor = TextProcessor("data/APRC/{}".format(args.data_file), sents_limit=args.sent_count)

    if args.dataset_random_every_time:
        train_dataset = DatasetRandom(text_as_list=text_processor.train_sents,
                                      tokenizer=text_processor.tokenizer,
                                      w2id=text_processor.w2id,
                                      max_seq_len=text_processor.max_seq_len,
                                      max_masked_size=text_processor.max_masked_size,
                                      to_cuda=args.to_cuda)

    else:
        train_dataset = DatasetConsistent(text_as_list=text_processor.train_sents,
                                          tokenizer=text_processor.tokenizer,
                                          w2id=text_processor.w2id,
                                          max_seq_len=text_processor.max_seq_len,
                                          max_masked_size=text_processor.max_masked_size,
                                          to_cuda=args.to_cuda)

    eval_dataset = DatasetConsistent(text_as_list=text_processor.eval_sents,
                                     tokenizer=text_processor.tokenizer,
                                     w2id=text_processor.w2id,
                                     max_seq_len=text_processor.max_seq_len,
                                     max_masked_size=text_processor.max_masked_size,
                                     to_cuda=args.to_cuda)

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


if __name__ == "__main__":
    main()
