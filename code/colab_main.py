import random
import argumenter
from logger import Logger
from plotter import Plotter
from model.cnp import CNP
from training.trainer import Trainer
from data_processing.dataset import DatasetNonContextual
from data_processing.text_processor import TextProcessor


def main():
    logger = Logger()
    logger.log("Starting CNP")

    args = argumenter.parse_arguments(logger)

    if abs(args.test_size - -1.0) < 0.01 and args.abs_test_size == -1:
        raise Exception("At least one of test size parameters must be set")
    if abs(args.test_size - -1.0) >= 0.01 and args.abs_test_size != -1:
        raise Exception("Only one of test size parameters should be set")

    test_size = args.abs_test_size
    if args.abs_test_size == -1:
        test_size = args.test_size

    random_seed = args.random_seed
    if not random_seed:
        random_seed = random.randint(1, 2_000_000_000)
    random.seed(a=random_seed)
    logger.log(f"Using seed={random_seed}")

    logger.log("Init Text Processor")
    text_processor = TextProcessor("data/APRC/{}".format(args.data_file),
                                   "data/embeddings/wiki-news-300d-1M.vec",
                                   test_size=test_size,
                                   sents_limit=args.sent_count,
                                   rare_word_threshold=args.rare_threshold,
                                   logger=logger)


    logger.log("Init Train Dataset")
    train_dataset = DatasetNonContextual(sents=text_processor.train_sents,
                                         max_seq_len=text_processor.max_seq_len,
                                         mask_ratios=args.train_mask_ratios,
                                         random_every_time=args.dataset_random_every_time,
                                         to_cuda=args.to_cuda)

    logger.log("Init Test Datasets")
    test_datasets = []
    test_datasets.append(DatasetNonContextual(sents=text_processor.test25,
                                              max_seq_len=text_processor.max_seq_len,
                                              mask_ratios=[.25],
                                              to_cuda=args.to_cuda))

    test_datasets.append(DatasetNonContextual(sents=text_processor.test50,
                                              max_seq_len=text_processor.max_seq_len,
                                              mask_ratios=[0.5],
                                              to_cuda=args.to_cuda))

    logger.log("Vocab size: ", len(text_processor.id2w))

    logger.log("Init model")
    model = CNP(enc_hidden_layers=args.enc_layers,
                dec_hidden_layers=args.dec_layers,
                emb_weight=text_processor.embedding_matrix,
                max_seq_len=text_processor.max_seq_len,
                use_weight_matrix=args.use_weight_matrix,
                dropout=args.dropout,
                use_latent=args.use_latent,
                nheads=args.number_of_heads,
                normalize_weights=args.normalize_weights,
                to_cuda=args.to_cuda)

    logger.log("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    logger.log("Init Trainer")

    # assume every test_ds has only one mask ration
    tags = [test_ds.mask_ratios[0] for test_ds in test_datasets]

    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      test_datasets=test_datasets,
                      tags=tags,
                      batch_size=args.batch_size,
                      opt=args.opt,
                      learning_rate=args.learning_rate,
                      momentum=args.momentum,
                      epoch_count=args.epochs,
                      acc_topk=args.topk,
                      kl_weight=args.kl_weight,
                      print_interval=args.print_interval,
                      bleu_sents=text_processor.bleu_sents,
                      to_cuda=args.to_cuda,
                      logger=logger,
                      id2w=text_processor.id2w)
    logger.log("Start training")
    train_loss_per_epoch, test_losses_per_epoch = trainer.run()
    plotter = Plotter(train_loss_per_epoch, test_losses_per_epoch, tags, logger)
    plotter.plot()


if __name__ == "__main__":
    main()
