import argparse


class InputArgument():
    def __init__(self, name, short_name, help_str, default_val, param_type=str, nargs=None, const=None):
        self.name = name
        self.inp_args = [f"-{short_name}", f"--{name}"]
        self.inp_kwargs = {
            "default": default_val,
            "help": help_str
        }
        if param_type != str:
            self.inp_kwargs["type"] = param_type
        if nargs is not None:
            self.inp_kwargs["nargs"] = nargs
        if const is not None:
            self.inp_kwargs["const"] = const


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


input_arguments = [
    InputArgument("data_file", "da", "data_file (default: APRC_new1.txt)", "APRC_new1.txt"),
    InputArgument("epochs", "e", "training epochs (default: 200)", 200, int),
    InputArgument("dataset_random_every_time", "ds",
                  "random mask every call for getitem or only at init (default: False)", False, str2bool, nargs="?",
                  const=True),
    InputArgument("learning_rate", "lr", "learning rate (default: .0005)", .0005, float),
    InputArgument("to_cuda", "c", "use cuda (default: True)", True, str2bool, nargs="?", const=True),
    InputArgument("sent_count", "sc", "sent count (default: no limit)", 0, int),
    InputArgument("batch_size", "bs", "batch_size (default: 50)", 50, int),
    InputArgument("topk", "topk", "topk (default: [1, 5, 10])", [1, 5, 10], int, nargs="+"),
    InputArgument("train_mask_ratios", "mr", "train_mask_ratios (default: [0.25, 0.5])", [0.25, 0.5], float, nargs="+"),
    InputArgument("momentum", "moment", "momentum (default: 0.9)", .9, float),
    InputArgument("test_size", "ts", "test_size (default: -1)", -1.0, float),
    InputArgument("abs_test_size", "ats", "test_size (default: -1)", -1, int),
    InputArgument("rare_threshold", "rt", "rare word threshold (default: 10)", 10, float),
    InputArgument("hidden_repr", "hr", "hidden_repr (default: 1000)", 1000, int),
    InputArgument("enc_layers", "encl", "enc_layers (default: [512, 768])", [512, 768], int, nargs="+"),
    InputArgument("dec_layers", "decl", "dec_layers (default: [768, 1024, 512])", [768, 1024, 512], int, nargs="+"),
    InputArgument("opt", "opt", "opt (default: \"ADAM\")", "ADAM"),
    InputArgument("dropout", "dp", "dropout (default: 0.1)", .1, float),
    InputArgument("print_interval", "pi", "print interval (default: 60sec)", 60, int),
    InputArgument("use_weight_matrix", "uwm", "Whether to multiply last layer by weight matrix (default: True)", True,
                  str2bool, nargs="?", const=True),
    InputArgument("use_pos_embedding", "upe", "Whether to use embeddings for positions (default: True)", True, str2bool,
                  nargs="?", const=True),
    InputArgument("number_of_heads", "nheads", "number of heads for attention (default: 2)", 2, int),
    InputArgument("normalize_weights", "nw", "Whether to normalize weight matrix (default: True)", True, str2bool,
                  nargs="?", const=True),
    InputArgument("use_latent", "ul", "Whether to use latent encoder (default: True)", True, str2bool,
                  nargs="?", const=True),
    InputArgument("kl_weight", "kl", "Weight of KL divergence in the training loss (default: 1.0)", 1.0, float),
    InputArgument("random_seed", "rs", "random_seed (default: randomly selected)", 0, int)
]


def parse_arguments(logger):
    parser = argparse.ArgumentParser()

    for input_argument in input_arguments:
        parser.add_argument(*input_argument.inp_args, **input_argument.inp_kwargs)

    args = parser.parse_args()

    logger.log()
    logger.log("Argument Values:")

    for input_argument in input_arguments:
        exec(f"inp_value=args.{input_argument.name}")
        exec(f"logger.log(input_argument.name + ': ' + str(inp_value))")
    logger.log()

    return args
