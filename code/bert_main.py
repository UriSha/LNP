import os
import random

import torch
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
from torch.utils.data import DataLoader
from bert_based.dataset_bert import DatasetBert
from data_processing.text_processor import TextProcessor
from logger import Logger
import torch.nn as nn


def main():
    to_cuda = torch.cuda.is_available()
    print("to_cuda:", to_cuda)
    train_mask_rations = [0.25, 0.5]
    test_size = 10000
    topk = [1, 5, 10]
    nheads = 2
    use_weight_matrix = True
    normalize_weights = True
    sequential = True
    big_bert = True

    dropout = 0
    epoch_count = 20
    random_every_time = True

    logger = Logger()
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    random.seed(a=539463084)

    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_new1.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_small_mock.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_new2.txt"),
                                   os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"),
                                   test_size=test_size,
                                   sents_limit=0,
                                   rare_word_threshold=10,
                                   logger=logger)

    if big_bert:
        pretrained_model_name_or_path = 'bert-large-uncased'
    else:
        pretrained_model_name_or_path = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    eval_datasets = []
    eval_datasets.append(DatasetBert(text_processor.test25,
                                     text_processor.max_seq_len,
                                     mask_ratios=[0.25],
                                     id2w=text_processor.id2w, tokenizer=tokenizer, to_cuda=to_cuda))
    eval_datasets.append(DatasetBert(text_processor.test50,
                                     text_processor.max_seq_len,
                                     mask_ratios=[0.5],
                                     id2w=text_processor.id2w, tokenizer=tokenizer, to_cuda=to_cuda))

    tags = [eval_ds.mask_ratios[0] for eval_ds in eval_datasets]

    # tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor = train_dataset[0]

    print("Vocab size: ", len(text_processor.id2w))
    
    model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)

    if to_cuda:
        model = model.cuda()

    print("Model has {} parameters".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model.eval()

    loss_function = nn.CrossEntropyLoss()

    eval_loaders = []
    for eval_dataset in eval_datasets:
        eval_loaders.append(DataLoader(dataset=eval_dataset, batch_size=1, shuffle=False))

    if sequential:
        for i, eval_loader in enumerate(eval_loaders):
            print(f"Evaluating: {tags[i]}")
            losses = []
            for tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor in eval_loader:
                tokens_tensor = tokens_tensor.squeeze(dim=0)
                segments_tensors = segments_tensors.squeeze(dim=0)
                indexed_masked_tokes_tensor = indexed_masked_tokes_tensor.squeeze(dim=0)
                positions_to_predict_tensor = positions_to_predict_tensor.squeeze(dim=0)
                # Predict all tokens
                with torch.no_grad():
                    for j in range(len(indexed_masked_tokes_tensor)):
                        predictions = model(tokens_tensor, segments_tensors)
                        cur_indexed_to_predict = positions_to_predict_tensor[j]
                        cur_token_id_to_predict = indexed_masked_tokes_tensor[j]
                        loss = loss_function(predictions[0, cur_indexed_to_predict].unsqueeze(dim=0), cur_token_id_to_predict.unsqueeze(dim=0))
                        losses.append(loss.item())
                        tokens_tensor[0, cur_indexed_to_predict] = torch.argmax(predictions[0, cur_indexed_to_predict]).item()

            avg_loss = sum(losses) / len(losses)
            print(f"Finished evaluating {tags[i]:.2f}, loss: {avg_loss:.2f}")
    else:
        for i, eval_loader in enumerate(eval_loaders):
            print(f"Evaluating: {tags[i]}")
            losses = []
            for tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor in eval_loader:
                tokens_tensor = tokens_tensor.squeeze(dim=0)
                segments_tensors = segments_tensors.squeeze(dim=0)
                indexed_masked_tokes_tensor = indexed_masked_tokes_tensor.squeeze(dim=0)
                positions_to_predict_tensor = positions_to_predict_tensor.squeeze(dim=0)
                with torch.no_grad():
                    predictions = model(tokens_tensor, segments_tensors)
                    for indexed_to_predict, token_id_to_predict in zip(positions_to_predict_tensor, indexed_masked_tokes_tensor):
                        loss = loss_function(predictions[0, indexed_to_predict].unsqueeze(dim=0), token_id_to_predict.unsqueeze(dim=0))
                        losses.append(loss.item())


            avg_loss = sum(losses) / len(losses)
            print(f"Finished evaluating {tags[i]:.2f}, loss: {avg_loss:.2f}")

if __name__ == "__main__":
    main()
