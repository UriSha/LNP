import argparse

# from data_processing.dataset_consistent import DatasetConsistent
# from data_processing.dataset_random import DatasetRandom
# from data_processing.text_processors.text_processor import TextProcessor
from data_processing.dataset_non_contextual import DatasetNonContextual
from data_processing.text_processors.text_processor_non_contextual import TextProcessorNonContextual
from model.cnp import CNP
from training import Trainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class InputArgument():
    def __init__(self, name, short_name, help_str, default_val, param_type=str, nargs=None, const=None):
        self.name = name
        self.inp_args = [f"-{short_name}", f"--{name}"]
        self.inp_kwargs = {
            "default" : default_val,
            "help" : help_str
        }
        if param_type != str:
            self.inp_kwargs["type"] = param_type
        if nargs is not None:
            self.inp_kwargs["nargs"] = nargs
        if const is not None:
            self.inp_kwargs["const"] = const


input_arguments = [
    InputArgument("data_file", "da", "data_file (default: APRC_new1.txt)", "APRC_new1.txt"),
    InputArgument("epochs", "e", "training epochs (default: 200)", 200, int),
    InputArgument("dataset_random_every_time", "ds", "random mask every call for getitem or only at init (default: False)", False, str2bool, nargs="?", const=True),
    InputArgument("learning_rate", "lr", "learning rate (default: .0005)", .0005, float),
    InputArgument("to_cuda", "c", "use cuda (default: True)", True, str2bool, nargs="?", const=True),
    InputArgument("sent_count", "sc", "sent count (default: no limit)", 0, int),
    InputArgument("batch_size", "bs", "batch_size (default: 50)", 50, int),
    InputArgument("mask_ratio", "mr", "max_ratio (default: 0.25)", .25, float),
    InputArgument("topk", "topk", "topk (default: 1)", 1, int),
    InputArgument("momentum", "moment", "momentum (default: 0.9)", .9, float),
    InputArgument("test_size", "ts", "test_size (default: 0.1)", .1, float),
    InputArgument("rare_threshold", "rt", "rare word threshold (default: 10)", 10, float),
    InputArgument("hidden_repr", "hr", "hidden_repr (default: 1000)", 1000, int),
    InputArgument("enc_layers", "encl", "enc_layers (default: [600, 600, 600, 600])", [600, 600, 600, 600], int, nargs="+"),
    InputArgument("dec_layers", "decl", "dec_layers (default: [600, 600, 600, 300])", [600, 600, 600, 300], int, nargs="+"),
    InputArgument("opt", "opt", "opt (default: \"ADAM\")", "ADAM"),
    InputArgument("dropout", "dp", "dropout (default: 0.1)", .1, float),
    InputArgument("print_interval", "pi", "print interval (default: 60sec)", 60, int),
    InputArgument("use_weight_matrix", "uwm", "Whether to multiply last layer by weight matrix (default: True)", True, str2bool, nargs="?", const=True),
    InputArgument("use_weight_loss", "uwl", "Whether to use weights for unbalanced data (default: False)", False, str2bool, nargs="?", const=True),
    InputArgument("use_pos_embedding", "upe", "Whether to use embeddings for positions (default: True)", True, str2bool, nargs="?", const=True),
    InputArgument("concat_embeddings", "ce", "Whether to concat sentence and position embeddings (default: False)", False, str2bool, nargs="?", const=True),
    InputArgument("use_attention", "attn", "Whether to use attention (default: True)", True, str2bool, nargs="?", const=True),
    InputArgument("number_of_heads", "nheads", "number of heads for attention (default: 2)", 2, int)
]


def parse_arguments():
    parser = argparse.ArgumentParser()

    for input_argument in input_arguments:
        parser.add_argument(*input_argument.inp_args, **input_argument.inp_kwargs)

    return parser.parse_args()


def main():
    print("Starting CNP")
    args = parse_arguments()

    print()
    print("Argument Values:")
    for input_argument in input_arguments:
        exec(f"inp_value=args.{input_argument.name}")
        exec(f"print(input_argument.name + ': ' + str(inp_value))")
    print()
    
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
                nheads=args.number_of_heads,
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
