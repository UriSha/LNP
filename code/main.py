from data_processing.dataset import dataset_random
from model.cnp import CNP
from training import Trainer


def read_data(path):
    with open(path, "r") as f:
        text = f.readlines()
    text = [sent.rstrip("\n").split(" ") for sent in text]
    return text


if __name__ == "__main__":
    sents = read_data("data/APRC/APRC_small_mock.txt")
    dataset = dataset_random(sents, to_cuda=False)
    model = CNP(769, 1, 800, [700], [700], len(dataset.id2w), dataset.max_seq_len, dataset.max_masked_size)
    trainer = Trainer(model, dataset, None, 2, 0.001, 3, False)
    trainer.run()