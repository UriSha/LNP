import argparse

# from data_processing.dataset_consistent import DatasetConsistent
# from data_processing.dataset_random import DatasetRandom
# from data_processing.text_processors.text_processor import TextProcessor
from data_processing.dataset_non_contextual import DatasetNonContextual
from data_processing.text_processors.text_processor_non_contextual import TextProcessorNonContextual
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
    parser.add_argument('-mr', '--mask_ratio',
                        help="max_ratio (default: 0.25)",
                        default=.25,
                        type=float)
    parser.add_argument('-topk', '--topk',
                        help="topk (default: 15)",
                        default=15,
                        type=int)
    parser.add_argument('-moment', '--momentum',
                        help="momentum (default: 0)",
                        default=0,
                        type=float)
    parser.add_argument('-ts', '--test_size',
                        help="test_size (default: 0.1)",
                        default=0.1,
                        type=float)
    parser.add_argument('-rt', '--rare_threshold',
                        help="rare word threshold (default: 10)",
                        default=10,
                        type=float)
    parser.add_argument('-hr', '--hidden_repr',
                        help="hidden_repr (default: 1000)",
                        default=1000,
                        type=int)
    parser.add_argument('-encl', '--enc_layers',
                        help="enc_layers (default: [2000, 1700, 1300])",
                        nargs="+",
                        type=int,
                        default=[2000, 1700, 1300])
    parser.add_argument('-decl', '--dec_layers',
                        help="dec_layers (default: [2000, 1700, 1300])",
                        nargs="+",
                        type=int,
                        default=[2000, 1700, 1300])
    parser.add_argument('-opt', '--opt',
                        help="opt (default: \"SGD\")",
                        default="SGD",
                        type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
           
    text_processor = TextProcessorNonContextual("data/APRC/{}".format(args.data_file),
                                                "data/embeddings/wiki-news-300d-1M.vec", 
                                                test_size=args.test_size,
                                                sents_limit=args.sent_count, 
                                                rare_word_threshold=args.rare_threshold)

    # if args.dataset_random_every_time:
    #     train_dataset = DatasetRandom(text_as_list=text_processor.train_sents,
    #                                   tokenizer=text_processor.tokenizer,
    #                                   w2id=text_processor.w2id,
    #                                   max_seq_len=text_processor.max_seq_len,
    #                                   max_masked_size=text_processor.max_masked_size,
    #                                   mask_ratio=args.mask_ratio,
    #                                   to_cuda=args.to_cuda)

    # else:
    #     train_dataset = DatasetConsistent(text_as_list=text_processor.train_sents,
    #                                       tokenizer=text_processor.tokenizer,
    #                                       w2id=text_processor.w2id,
    #                                       max_seq_len=text_processor.max_seq_len,
    #                                       max_masked_size=text_processor.max_masked_size,
    #                                       mask_ratio=args.mask_ratio,
    #                                       to_cuda=args.to_cuda)

    train_dataset = DatasetNonContextual(text_as_list=text_processor.train_sents,
                                     w2id=text_processor.w2id,
                                     id2w=text_processor.id2w,
                                     max_seq_len=text_processor.max_seq_len,
                                     max_masked_size=text_processor.max_masked_size,
                                     mask_ratio=args.mask_ratio,
                                     to_cuda=args.to_cuda)

    eval_dataset = DatasetNonContextual(text_as_list=text_processor.eval_sents,
                                     w2id=text_processor.w2id,
                                     id2w=text_processor.id2w,
                                     max_seq_len=text_processor.max_seq_len,
                                     max_masked_size=text_processor.max_masked_size,
                                     mask_ratio=args.mask_ratio,
                                     to_cuda=args.to_cuda)

    print("Vocab size: ", len(text_processor.id2w))
    model = CNP(embedding_size=text_processor.vec_size,
                hidden_repr=args.hidden_repr,
                enc_hidden_layers=args.enc_layers,
                dec_hidden_layers=args.dec_layers,
                output_size=len(text_processor.id2w),
                max_target_size=text_processor.max_masked_size,
                w2id = text_processor.w2id,
                id2w = text_processor.id2w,
                emb_weight = text_processor.embed_matrix,
                max_seq_len=text_processor.max_seq_len,
                padding_idx = text_processor.pad_index,
                to_cuda=args.to_cuda)

    trainer = Trainer(model=model,
                      training_dataset=train_dataset,
                      evaluation_dataset=eval_dataset,
                      batch_size=args.batch_size,
                      opt=args.opt,
                      learning_rate=args.learning_rate,
                      momentum=args.momentum,
                      epoch_count=args.epochs,
                      acc_topk=args.topk,
                      to_cuda=args.to_cuda)
    trainer.run()


if __name__ == "__main__":
    main()
