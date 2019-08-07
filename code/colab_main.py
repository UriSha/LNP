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
                        help="batch_size (default: 50)",
                        default=50,
                        type=int)
    parser.add_argument('-mr', '--mask_ratio',
                        help="max_ratio (default: 0.25)",
                        default=.25,
                        type=float)
    parser.add_argument('-topk', '--topk',
                        help="topk (default: 5)",
                        default=5,
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
                        help="enc_layers (default: [600, 600, 600, 600])",
                        nargs="+",
                        type=int,
                        default=[600, 600, 600, 600])
    parser.add_argument('-decl', '--dec_layers',
                        help="dec_layers (default: [600, 600, 600, 300])",
                        nargs="+",
                        type=int,
                        default=[600, 600, 600, 300])
    parser.add_argument('-opt', '--opt',
                        help="opt (default: \"ADAM\")",
                        default="ADAM",
                        type=str)
    parser.add_argument('-dp', '--dropout',
                        help="dropout (default: 0.1)",
                        default=0.1,
                        type=float)
    parser.add_argument('-pi', '--print_interval',
                        help="print interval (default: 60sec)",
                        default=60,
                        type=int)
    parser.add_argument('-uwm', '--use_weight_matrix',
                        help="Whether to multiply last layer by weight matrix (default: True)",
                        default=True,
                        type=bool)
    parser.add_argument('-uwl', '--use_weight_loss',
                        help="Whether to use weights for unbalanced data (default: False)",
                        default=False,
                        type=bool)
    parser.add_argument('-upe', '--use_pos_embedding',
                        help="Whether to use embeddings for positions (default: True)",
                        default=True,
                        type=bool)
    parser.add_argument('-ce', '--concat_embeddings',
                        help="Whether to concat sentence and position embeddings (default: False)",
                        default=False,
                        type=bool)
    parser.add_argument('-attn', '--use_attention',
                        help="Whether to use attention (default: True)",
                        default=True,
                        type=bool)
    return parser.parse_args()


def main():
    print("Starting CNP")
    args = parse_arguments()

    print(f"use_weight_matrix: {args.use_weight_matrix}")
    print(f"use_weight_loss: {args.use_weight_loss}")
    print(f"use_pos_embedding: {args.use_pos_embedding}")
    print(f"concat_embeddings: {args.concat_embeddings}")
    print(f"use_attention: {args.use_attention}")

    print("Init text processor")
    text_processor = TextProcessorNonContextual("data/APRC/{}".format(args.data_file),
                                                "data/embeddings/wiki-news-300d-1M.vec",
                                                test_size=args.test_size,
                                                sents_limit=args.sent_count,
                                                rare_word_threshold=args.rare_threshold,
                                                use_weight_loss = args.use_weight_loss)

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

    print("Init train Dataset")
    train_dataset = DatasetNonContextual(text_as_list=text_processor.train_sents,
                                         w2id=text_processor.w2id,
                                         id2w=text_processor.id2w,
                                         max_seq_len=text_processor.max_seq_len,
                                         max_masked_size=text_processor.max_masked_size,
                                         mask_ratio=args.mask_ratio,
                                         to_cuda=args.to_cuda)

    print("Init test Dataset")
    eval_dataset = DatasetNonContextual(text_as_list=text_processor.eval_sents,
                                        w2id=text_processor.w2id,
                                        id2w=text_processor.id2w,
                                        max_seq_len=text_processor.max_seq_len,
                                        max_masked_size=text_processor.max_masked_size,
                                        mask_ratio=args.mask_ratio,
                                        to_cuda=args.to_cuda)

    print("Vocab size: ", len(text_processor.id2w))
    print("Init model")
    model = CNP(embedding_size=text_processor.vec_size,
                hidden_repr=args.hidden_repr,
                enc_hidden_layers=args.enc_layers,
                dec_hidden_layers=args.dec_layers,
                max_target_size=text_processor.max_masked_size,
                w2id=text_processor.w2id,
                id2w=text_processor.id2w,
                emb_weight=text_processor.embed_matrix,
                max_seq_len=text_processor.max_seq_len,
                padding_idx=text_processor.pad_index,
                use_weight_matrix = args.use_weight_matrix,
                dropout=args.dropout,
                use_pos_embedding = args.use_pos_embedding,
                attn = args.use_attention,
                concat_embeddings=args.concat_embeddings,
                to_cuda=args.to_cuda)

    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print("Init Trainer")
    trainer = Trainer(model=model,
                      training_dataset=train_dataset,
                      evaluation_dataset=eval_dataset,
                      batch_size=args.batch_size,
                      opt=args.opt,
                      learning_rate=args.learning_rate,
                      momentum=args.momentum,
                      epoch_count=args.epochs,
                      acc_topk=args.topk,
                      print_interval = args.print_interval,
                      word_weights = text_processor.word_weights,
                      use_weight_loss = args.use_weight_loss,
                      to_cuda=args.to_cuda)
    print("Start training")
    trainer.run()


if __name__ == "__main__":
    main()
