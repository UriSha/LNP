from data_processing.dataset_consistent import DatasetConsistent

from data_processing.text_processor import TextProcessor
from model.cnp import CNP
from training import Trainer


def main():
    to_cuda = False

    text_processor = TextProcessor("data/APRC/APRC_small_mock.txt", sents_limit=1000)
    train_dataset = DatasetConsistent(text_as_list=text_processor.train_sents,
                                          tokenizer=text_processor.tokenizer,
                                          w2id=text_processor.w2id,
                                          max_seq_len=text_processor.max_seq_len,
                                          max_masked_size=text_processor.max_masked_size,
                                          to_cuda=to_cuda)
    eval_dataset = DatasetConsistent(text_as_list=text_processor.eval_sents,
                                          tokenizer=text_processor.tokenizer,
                                          w2id=text_processor.w2id,
                                          max_seq_len=text_processor.max_seq_len,
                                          max_masked_size=text_processor.max_masked_size,
                                          to_cuda=to_cuda)
    model = CNP(context_size=769,
                target_size=1,
                hidden_repr=800,
                enc_hidden_layers=[800, 800],
                dec_hidden_layers=[850, 1000],
                output_size=len(text_processor.id2w),
                max_sent_len=text_processor.max_seq_len,
                max_target_size=text_processor.max_masked_size,
                to_cuda=to_cuda)
    trainer = Trainer(model, train_dataset, eval_dataset, 16, 0.005, 100, to_cuda)
    trainer.run()


if __name__ == "__main__":
    main()
