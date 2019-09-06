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
from training.bleu import corpus_bleu_with_joint_refrences
from nltk.translate.bleu_score import corpus_bleu



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bm', '--bert_fine_tuned_path',
                        help="bert_fine_tuned_path",
                        default='bert_finetuned/pytorch_model.bin')
    parser.add_argument('-se', '--sequential',
                        help="sequential (default: True)",
                        default="False")
    parser.add_argument('-sb', '--use_small_bert',
                        help="sequential (default: True)",
                        default="True")
    parser.add_argument('-pw', '--print_w',
                        help="print_w (default: True)",
                        default="False")
    args = parser.parse_args()
    to_cuda = torch.cuda.is_available()


    bert_fine_tuned_path = args.bert_fine_tuned_path
    if bert_fine_tuned_path == "None":
        bert_fine_tuned_path = None
   # bert_fine_tuned_path = None
    test_size = 10000
    sequential = args.sequential if args.sequential == "True" else False
    small_bert = args.use_small_bert if args.use_small_bert == "True" else False
    print_w = args.print_w if args.print_w == "True" else False

    print("to_cuda:", to_cuda)
    print("bert_fine_tuned_path", bert_fine_tuned_path)
    print("sequential", sequential)
    print("small_bert", small_bert)
    print("print_w", print_w)


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
                                   rare_word_threshold=0,
                                   logger=logger)

    # text_processor = TextProcessor(os.path.join(cur_dir, "../data/APRC/APRC_small_mock1.txt"),
    #                                os.path.join(cur_dir, "../data/embeddings/small_fasttext.txt"),
    #                                test_size=test_size,
    #                                sents_limit=0,
    #                                rare_word_threshold=0,
    #                                logger=logger)

    if small_bert:
        print("will use bert base")
        pretrained_model_name_or_path = 'bert-base-uncased'
    else:
        print("will use bert large")
        pretrained_model_name_or_path = 'bert-large-uncased'

    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)

    blue_sents = text_processor.bleu_sents
    bert_bleu_sents = []
    for sent in blue_sents:
        tokenized = tokenizer.tokenize(" ".join(sent))
        bert_bleu_sents.append(tokenized)

    print("len(blue_sents): ", len(bert_bleu_sents))


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

    if print_w:
        print("bert weights")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    print()
    if bert_fine_tuned_path is not None:
        print("starting to load pre-trained bert")
        bert_fine_tuned_state_dict = torch.load(bert_fine_tuned_path)

        if print_w:
            print("bert_fine_tuned_state_dict:")
            for param_tensor in bert_fine_tuned_state_dict:
                print(param_tensor, "\t", bert_fine_tuned_state_dict[param_tensor].size())

        print("removing cls.seq_relationship.weight and cls.seq_relationship.bias from bert_fine_tuned_state_dict")
        del bert_fine_tuned_state_dict["cls.seq_relationship.weight"]
        del bert_fine_tuned_state_dict["cls.seq_relationship.bias"]

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

    golden_sents = []
    predicted_sents = []
    if sequential:
        for i, eval_loader in enumerate(eval_loaders):
            print(f"Evaluating: {tags[i]}")
            golden_sents.append([])
            predicted_sents.append([])
            losses = []
            for tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor in eval_loader:
                golden_sent = list(map(int, tokens_tensor.clone().squeeze()))
                predicted_sent = list(map(int, tokens_tensor.clone().squeeze()))

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

                        golden_sent[cur_indexed_to_predict] = cur_token_id_to_predict.item()
                        predicted_sent[cur_indexed_to_predict] = torch.argmax(predictions[0, cur_indexed_to_predict]).item()

                    golden_sent = tokenizer.convert_ids_to_tokens(golden_sent[1:-1])
                    predicted_sent = tokenizer.convert_ids_to_tokens(predicted_sent[1:-1])

                    golden_sents[i].append([golden_sent])
                    predicted_sents[i].append(predicted_sent)

            avg_loss = sum(losses) / len(losses)
            print(f"Finished evaluating {tags[i]:.2f}, loss: {avg_loss:.2f}")
    else:
        for i, eval_loader in enumerate(eval_loaders):
            print(f"Evaluating: {tags[i]}")
            golden_sents.append([])
            predicted_sents.append([])
            losses = []
            for tokens_tensor, segments_tensors, indexed_masked_tokes_tensor, positions_to_predict_tensor in eval_loader:
                golden_sent = list(map(int,tokens_tensor.clone().squeeze()))
                predicted_sent = list(map(int,tokens_tensor.clone().squeeze()))

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

                        golden_sent[indexed_to_predict] = token_id_to_predict.item()
                        predicted_sent[indexed_to_predict] = torch.argmax(predictions[0, indexed_to_predict]).item()

                    golden_sent = tokenizer.convert_ids_to_tokens(golden_sent[1:-1])
                    predicted_sent = tokenizer.convert_ids_to_tokens(predicted_sent[1:-1])

                    golden_sents[i].append([golden_sent])
                    predicted_sents[i].append(predicted_sent)

            # total loss
            avg_loss = sum(losses) / len(losses)
            print(f"Finished evaluating {tags[i]:.2f}, loss: {avg_loss:.2f}")

    for i, eval_loader in enumerate(eval_loaders):
        # total bleu
        print(f"calculating bleu for {tags[i]:.2f}")
        blue_with_only_golden = corpus_bleu(golden_sents[i],predicted_sents[i])
        print(f"blue_with_only_golden for {tags[i]:.2f}:", blue_with_only_golden)

        bleu_score = corpus_bleu_with_joint_refrences(bert_bleu_sents,golden_sents[i],predicted_sents[i])
        print(f"bleu_score with {len(bert_bleu_sents)} sents from eval for {tags[i]:.2f}:", bleu_score)

if __name__ == "__main__":
    main()