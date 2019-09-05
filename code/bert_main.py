import os
import random

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertForMaskedLM, BertTokenizer
from torch.utils.data import DataLoader
import argparse
from bert_based.dataset_bert import DatasetBert
from data_processing.text_processor import TextProcessor
from logger import Logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bm', '--bert_fine_tuned_path',
                        help="bert_fine_tuned_path",
                        default='bert_based/bert_finetuned/pytorch_model.bin')
    parser.add_argument('-se', '--sequential',
                        help="sequential (default: True)",
                        default="True")
    parser.add_argument('-sb', '--use_small_bert',
                        help="sequential (default: True)",
                        default="True")
    args = parser.parse_args()
    to_cuda = torch.cuda.is_available()

    bert_fine_tuned_path = args.bert_fine_tuned_path
    train_mask_rations = [0.25, 0.5]
    test_size = 10000
    topk = [1, 5, 10]
    nheads = 2
    use_weight_matrix = True
    normalize_weights = True
    sequential = args.sequential if args.sequential == "True" else False
    small_bert = args.use_small_bert if  args.use_small_bert == "True" else False

    dropout = 0
    epoch_count = 20
    random_every_time = True

    print("to_cuda:", to_cuda)
    print("bert_fine_tuned_path", bert_fine_tuned_path)
    print("sequential", sequential)
    print("small_bert", small_bert)


    logger = Logger()
    cur_dir = os.path.dirname(os.path.realpath(__file__))

    random.seed(a=539463084)

    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_new1.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_small_mock.txt"),
    #                                             os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"), test_size=test_size, mask_ratio=mask_ratio,
    #                                             sents_limit=10000, rare_word_threshold=1, use_weight_loss=True)
    print("init text_processor")
    text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_new2.txt"),
                                   os.path.join(cur_dir, "../data/embeddings/wiki-news-300d-1M.vec"),
                                   test_size=test_size,
                                   sents_limit=0,
                                   rare_word_threshold=10,
                                   logger=logger)

    if small_bert:
        print("will use bert base")
        pretrained_model_name_or_path = 'bert-base-uncased'
    else:
        print("will use bert large")
        pretrained_model_name_or_path = 'bert-large-uncased'

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

    print("Vocab size: ", len(text_processor.id2w))

    model = BertForMaskedLM.from_pretrained(pretrained_model_name_or_path)

    print("bert weights")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()
    if bert_fine_tuned_path is not None:
        print("starting to load pre-trained bert")
        print("bert_fine_tuned_state_dict:")
        bert_fine_tuned_state_dict = torch.load(bert_fine_tuned_path)
        for param_tensor in bert_fine_tuned_state_dict:
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        model.load_state_dict(bert_fine_tuned_state_dict)
        print("loaded pre-trained bert")

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
                        loss = loss_function(predictions[0, cur_indexed_to_predict].unsqueeze(dim=0),
                                             cur_token_id_to_predict.unsqueeze(dim=0))
                        losses.append(loss.item())
                        tokens_tensor[0, cur_indexed_to_predict] = torch.argmax(
                            predictions[0, cur_indexed_to_predict]).item()

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
                    for indexed_to_predict, token_id_to_predict in zip(positions_to_predict_tensor,
                                                                       indexed_masked_tokes_tensor):
                        loss = loss_function(predictions[0, indexed_to_predict].unsqueeze(dim=0),
                                             token_id_to_predict.unsqueeze(dim=0))
                        losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            print(f"Finished evaluating {tags[i]:.2f}, loss: {avg_loss:.2f}")


if __name__ == "__main__":
    main()
