from model.cnp import CNP
from training import Trainer
from data_processing.dataset_once_random import dataset_consistent
from data_processing.text_processor import TextProcessor


if __name__ == "__main__":
    text_processor = TextProcessor("data/APRC/APRC_small_mock.txt", sents_limit=1000)
    train_dataset = dataset_consistent(text_processor.train_sents, to_cuda=False)
    eval_dataset = dataset_consistent(text_processor.eval_sents, to_cuda=False)
    model = CNP(context_size=769, 
            target_size=1, 
            hidden_repr=800, 
            enc_hidden_layers=[800, 800], 
            dec_hidden_layers=[850, 1000], 
            output_size=len(text_processor.id2w), 
            max_sent_len=text_processor.max_seq_len, 
            max_target_size=text_processor.max_masked_size,
            to_cuda=args.to_cuda)
    trainer = Trainer(model, train_dataset, eval_dataset, 16, 0.005, 100, False)
    trainer.run()