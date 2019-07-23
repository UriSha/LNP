from data_processing.dataset_consistent import DatasetConsistent
from data_processing.text_processors.text_processor import TextProcessor
from data_processing.dataset_non_contextual import DatasetNonContextual
from data_processing.text_processors.text_processor_non_contextual import TextProcessorNonContextual
from model.cnp import CNP
from training import Trainer


def main():
    to_cuda = False
    mask_ratio = 0.1

    text_processor = TextProcessorNonContextual("data/APRC/APRC_new1.txt",
                                                "data/embeddings/wiki-news-300d-1M.vec", test_size=0.1,
                                                sents_limit=10000, rare_word_threshold=10)
    # text_processor = TextProcessor("data/APRC/APRC_small_mock.txt", test_size=0.1, sents_limit=500)
    # text_processor = TextProcessor("data/APRC/APRC_small_mock.txt", test_size=0.05, sents_limit=5)
    train_dataset = DatasetNonContextual(text_processor.train_sents, text_processor.w2id, text_processor.id2w,
                                text_processor.max_seq_len, text_processor.max_masked_size, mask_ratio=mask_ratio, to_cuda=to_cuda)
    eval_dataset = DatasetNonContextual(text_processor.eval_sents, text_processor.w2id, text_processor.id2w,
                                text_processor.max_seq_len, text_processor.max_masked_size, mask_ratio=mask_ratio, to_cuda=to_cuda)
    # train_dataset = DatasetConsistent(text_as_list=text_processor.train_sents,
    #                                   tokenizer=text_processor.tokenizer,
    #                                   w2id=text_processor.w2id,
    #                                   max_seq_len=text_processor.max_seq_len,
    #                                   max_masked_size=text_processor.max_masked_size,
    #                                   mask_ratio=mask_ratio,
    #                                   to_cuda=to_cuda)
    # eval_dataset = DatasetConsistent(text_as_list=text_processor.eval_sents,
    #                                  tokenizer=text_processor.tokenizer,
    #                                  w2id=text_processor.w2id,
    #                                  max_seq_len=text_processor.max_seq_len,
    #                                  max_masked_size=text_processor.max_masked_size,
    #                                  mask_ratio=mask_ratio,
    #                                  to_cuda=to_cuda)

    print("Vocab size: ", len(text_processor.id2w))
    model = CNP(embedding_size=text_processor.vec_size,
                hidden_repr=1024,
                enc_hidden_layers=[800, 1000],
                dec_hidden_layers=[768, 1024, 2048],
                output_size=len(text_processor.id2w),
                max_target_size=text_processor.max_masked_size,
                w2id = text_processor.w2id,
                id2w = text_processor.id2w,
                emb_weight = text_processor.embed_matrix,
                padding_idx = text_processor.pad_index,
                to_cuda=to_cuda)
    trainer = Trainer(model=model,
                      training_dataset=train_dataset,
                      evaluation_dataset=eval_dataset,
                      batch_size=32,
                      opt="SGD",
                      learning_rate=0.01,
                      momentum=0,
                      epoch_count=200,
                      acc_topk=15,
                      to_cuda=to_cuda)
    trainer.run()


if __name__ == "__main__":
    main()
