import os
import time
from data_processing.dataset_non_contextual import DatasetNonContextual
from data_processing.text_processors.text_processor_non_contextual import TextProcessorNonContextual
from model.cnp import CNP
from training import Trainer
from plotter import Plotter


def main():
    to_cuda = False
    attn = False
    mask_ratio = 0.25
    test_size = 0.1
    topk = 1
    nheads = 2
    use_weight_loss = False
    use_weight_matrix = True
    use_pos_embedding = True
    concat_embeddings = False
    normalize_weights = True
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    # text_processor = TextProcessorNonContextual(os.path.join(cur_dir, "../data/APRC/APRC_new1.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    # text_processor = TextProcessorNonContextual(os.path.join(cur_dir, "../data/APRC/APRC_small_mock.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    text_processor = TextProcessorNonContextual(os.path.join(cur_dir, "../data/APRC/APRC_small_mock.txt"),
                                                os.path.join(cur_dir, "../data/embeddings/small_fasttext.txt"), test_size=test_size, mask_ratio=mask_ratio,
                                                sents_limit=10000, rare_word_threshold=0, use_weight_loss=use_weight_loss)
                                                
    train_dataset = DatasetNonContextual(text_processor.train_sents, text_processor.w2id, text_processor.id2w,
                                         text_processor.max_seq_len, text_processor.max_masked_size,
                                         mask_ratio=mask_ratio, to_cuda=to_cuda)
    eval_dataset = DatasetNonContextual(text_processor.eval_sents, text_processor.w2id, text_processor.id2w,
                                        text_processor.max_seq_len, text_processor.max_masked_size,
                                        mask_ratio=mask_ratio, to_cuda=to_cuda)

    files_timestamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    print("Vocab size: ", len(text_processor.id2w))
    model = CNP(embedding_size=text_processor.vec_size,
                hidden_repr=300,
                enc_hidden_layers=[512, 768],
                dec_hidden_layers=[768, 1024, 512],
                max_target_size=text_processor.max_masked_size,
                w2id = text_processor.w2id,
                id2w = text_processor.id2w,
                emb_weight = text_processor.embed_matrix,
                max_seq_len = text_processor.max_seq_len,
                padding_idx = text_processor.pad_index,
                use_weight_matrix = use_weight_matrix,
                dropout=0,
                attn=attn,
                nheads=nheads,
                use_pos_embedding=use_pos_embedding,
                concat_embeddings=concat_embeddings,
                normalize_weights=normalize_weights,
                to_cuda=to_cuda)
    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # print(list(model.decoder.parameters()))
    trainer = Trainer(model=model,
                      training_dataset=train_dataset,
                      evaluation_dataset=eval_dataset,
                      batch_size=16,
                      opt="ADAM",
                      learning_rate=0.001,
                      momentum=0.9,
                      epoch_count=200,
                      acc_topk=topk,
                      print_interval=1,
                      word_weights = text_processor.word_weights,
                      use_weight_loss=use_weight_loss,
                      to_cuda=to_cuda,
                      files_timestamp=files_timestamp)
    train_loss, eval_loss = trainer.run()
    plotter = Plotter(train_loss, eval_loss, files_timestamp)
    plotter.plot()


if __name__ == "__main__":
    main()
