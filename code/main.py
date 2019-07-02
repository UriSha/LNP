import pickle as cPickle
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging


def load_data_and_embeddings():
    name = 'APRC'

    mid25 = cPickle.load(open('../data/%s/25_random.pkl' % name, 'rb'))
    index = cPickle.load(open('../data/%s/index.pkl' % name, 'rb'))
    w2id, id2w = cPickle.load(open('../data/%s/w2id_id2w.pkl' % name, 'rb'))

    with open('../data/embeddings/wiki-news-300d-1M.vec', encoding='utf-8') as fin:
        lines_embeddings = fin.readlines()
        print(len(lines_embeddings))

    # word_with_embeddings_vector = set([lines_embeddings[i].split(' ')[0] for i in range(1, len(lines_embeddings))])
    # print(len(word_with_embeddings_vector))

    # print("'s" in word_with_embeddings_vector)
    # data_mid25 = [[id2w[id] for id in sent[0]] for sent in mid25]

    # data_index = [[id2w[id] for id in sent] for sent in index]
    # all_words = set()
    # for sent in data_index:
    #     for w in sent:
    #         if w != "<EOS>":
    #             all_words.add(w)


    break_piont_here = True


def load_bert():
    logging.basicConfig(level=logging.INFO)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    # text = "[CLS] Who was Jim Henson ? Jim Henson was a puppeteer"
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a',
                              'puppet', '##eer', '[SEP]']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    tokenized_text_length = len(tokenized_text)
    tokens_tensor_shape = tokens_tensor.shape
    segments_tensors = segments_tensors.shape

    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # If you have a GPU, put everything on cuda
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12

    break_piont_here = True


def generage_original_text_from_index_and_id2w():
    import pickle as cPickle

    data_folder = "../data/APRC/"

    index = cPickle.load(open(data_folder + 'index.pkl', 'rb'))
    w2id, id2w = cPickle.load(open(data_folder + 'w2id_id2w.pkl', 'rb'))
    max_len = 0
    with open(data_folder + "APRC.txt", "w+") as f:
        for idx_sent in index:
            ### get original sentance
            max_len = max(max_len, len(idx_sent))
            original_sent = " ".join([id2w[idx_sent[i]] for i in range(len(idx_sent) - 1)]) + "\n"

            ### write sentance to file
            f.write(original_sent)
    print(max_len)

if __name__ == "__main__":
    print('Hello World!')
    load_data_and_embeddings()
    # load_bert()
    # generage_original_text_from_index_and_id2w()
    # t1 = torch.tensor([[[11, 12, 13, 14], [1, 2, 3, 4]]]).float()
    # t2 = torch.tensor([[[0, 0, 1, 1], [2, 2, 3, 3]]]).float()
    # t3 = torch.tensor([[[3, 2, 1, 0], [3, 2, 1, 0]]]).float()
    # print(t1.shape)
    # lst = [t1, t2, t3]
    #
    # average_transform(lst)
