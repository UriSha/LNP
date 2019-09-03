import os
import random

import torch
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer

from bert_based.dataset_bert import DatasetBert
from data_processing.text_processor import TextProcessor
from logger import Logger
import torch.nn as nn


def main():
    to_cuda = False
    train_mask_rations = [0.25, 0.5]
    test_size = 0.5
    topk = [1, 5, 10]
    nheads = 2
    use_weight_matrix = True
    normalize_weights = True

    dropout = 0
    epoch_count = 20
    random_every_time = True

    logger = Logger()
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    random.seed(a=5)

    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_new1.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_small_mock.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_small_mock1.txt"),
                                   os.path.join(cur_dir, "../data/embeddings/small_fasttext.txt"),
                                   test_size=test_size,
                                   sents_limit=10000,
                                   rare_word_threshold=0,
                                   logger=logger)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = DatasetBert(text_processor.train_sents,
                                text_processor.max_seq_len,
                                mask_ratios=train_mask_rations,
                                id2w=text_processor.id2w,
                                w2id=text_processor.w2id,
                                tokenizer=tokenizer,
                                random_every_time=random_every_time,
                                to_cuda=to_cuda)
    eval_datasets = []
    eval_datasets.append(DatasetBert(text_processor.test25,
                                     text_processor.max_seq_len,
                                     mask_ratios=[0.25],
                                     id2w=text_processor.id2w, w2id=text_processor.w2id, tokenizer=tokenizer, to_cuda=to_cuda))
    eval_datasets.append(DatasetBert(text_processor.test50,
                                     text_processor.max_seq_len,
                                     mask_ratios=[0.5],
                                     id2w=text_processor.id2w, w2id=text_processor.w2id, tokenizer=tokenizer, to_cuda=to_cuda))

    tags = [eval_ds.mask_ratios[0] for eval_ds in eval_datasets]

    tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor = train_dataset[0]

    print("Vocab size: ", len(text_processor.id2w))
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    for indexed_to_predict, token_id_to_predict in zip(positions_to_predict_tensor, indexed_masked_tokes_tensor):
        predicted_index = torch.argmax(predictions[0, indexed_to_predict]).item()
        predicted_token = train_dataset.tokenizer.convert_ids_to_tokens([predicted_index])[0]
        loss = loss_function(predictions[0,indexed_to_predict].unsqueeze(dim=0), token_id_to_predict.unsqueeze(dim=0))
        print(f"loss: {loss.item()}")

    print(predicted_token)
    x=4

    # print(list(model.decoder.parameters()))
    # trainer = Trainer(model=model,
    #                   train_dataset=train_dataset,
    #                   test_datasets=eval_datasets,
    #                   tags=tags,
    #                   batch_size=70,
    #                   opt="ADAM",
    #                   learning_rate=0.001,
    #                   momentum=0.9,
    #                   epoch_count=epoch_count,
    #                   acc_topk=topk,
    #                   print_interval=1,
    #                   bleu_sents=text_processor.bleu_sents,
    #                   to_cuda=to_cuda,
    #                   logger=logger,
    #                   id2w=text_processor.id2w)
    # train_loss, test_losses = trainer.run()
    # plotter = Plotter(train_loss, test_losses, tags, logger)
    # plotter.plot()


if __name__ == "__main__":
    main()
