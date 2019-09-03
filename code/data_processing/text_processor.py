import re
import torch
import random
from collections import defaultdict


class TextProcessor():
    def __init__(self, text_file_path, embed_file_path, test_size, rare_word_threshold, logger, sents_limit=None):

        self.logger = logger
        self.id2w = {}
        self.w2id = {}
        self.max_seq_len = 0
        self.sents = None
        self.embedding_matrix = None

        raw_sents = self.read_data(text_file_path, sents_limit)
        self.initiate_vocab(raw_sents, rare_word_threshold, embed_file_path)

        shuffled_sents = random.sample(self.sents, k=len(self.sents))
        if isinstance(test_size, float):
            test_size = int(len(self.sents) * test_size)
        self.train_sents = shuffled_sents[:-test_size]
        self.test_sents = shuffled_sents[-test_size:]
        sampled_bleu_sents = random.sample(self.test_sents, k=min(10000, len(self.test_sents)))
        self.bleu_sents = [[self.id2w[word_id] for word_id in bleu_sent] for bleu_sent in sampled_bleu_sents]

        test_size = len(self.test_sents) // 2

        self.test25 = self.test_sents[:test_size]
        self.test50 = self.test_sents[test_size:]


    def read_data(self, path, sents_limit):
        with open(path, "r") as f:
            if sents_limit:
                i = 0
                sent = next(f, None)
                text = []
                while sent and i < sents_limit:
                    i += 1
                    text.append(sent)
                    sent = next(f, None)
            else:
                text = f.readlines()

        return [sent.rstrip("\n").split(" ") for sent in text]


    def normalize_word(self, w):
        w = w.replace("`", "")
        if not bool(re.search(r'^\'\w', w)):
            w = w.replace("'", "")
        lst = re.findall(r'\'?\w+|\.|\,', w)
        if len(lst) == 3:
            if lst[1] == '.':
                if len(lst[2]) == 1:
                    if lst[2] not in ['i', 'a']:
                        return lst[:2]
        return lst


    def initiate_vocab(self, sents, rare_word_threshold, embed_file_path):
        max_len = 0
        new_sents = []
        w2cnt = defaultdict(int)

        temp_w2id, temp_id2w, embed_dict, vec_size, embeddings_file_lines = self.read_embeddings(embed_file_path)

        for sent in sents:
            new_sent = []
            for w in sent:
                w_list = self.normalize_word(w)
                if not isinstance(w_list, list):
                    w_list = [w_list]

                for word in w_list:
                    if word not in temp_w2id:  # i.e word has not embedding vector
                        word = "<UNK>"

                    new_sent.append(word)
                    w2cnt[word] += 1

            new_sents.append(new_sent)
            max_len = max(max_len, len(new_sent))

        rare_words_count = 0
        for k in w2cnt.keys():
            if w2cnt[k] < rare_word_threshold and k != "<UNK>" and k != "<PAD>":
                rare_words_count += 1
                word_id = temp_w2id[k]
                try:
                    del temp_w2id[k]
                    del temp_id2w[word_id]
                    del embed_dict[word_id]
                except KeyError:
                    self.logger.log("k = ", k)
                    self.logger.log("word_id = ", word_id)


        embed_list = [torch.zeros(vec_size)]
        new_w2id = {}
        new_w2id["<PAD>"] = 0
        new_id2w = {}
        new_id2w[0] = "<PAD>"
        for sent in new_sents:
            for i, word in enumerate(sent):
                if word not in temp_w2id:
                    sent[i] = "<UNK>"
                    word = "<UNK>"
                    if "<UNK>" not in new_w2id:
                        new_w2id["<UNK>"] = len(embed_list)
                        new_id2w[len(embed_list)] = "<UNK>"
                        embed_list.append(torch.zeros(vec_size))
                elif word not in new_w2id:
                    old_word_id = temp_w2id[word]
                    new_word_id = len(embed_list)
                    embed_list.append(self.line_to_embedding(embed_dict[old_word_id], vec_size, embeddings_file_lines))
                    new_w2id[word] = new_word_id
                    new_id2w[new_word_id] = word

        ratio = rare_words_count / len(w2cnt)
        self.logger.log(f"With rare_word_threshold = {rare_word_threshold}, the ratio of rare words (that were removed) is: {ratio}")

        self.embedding_matrix = torch.stack(embed_list)
        self.id2w = new_id2w
        self.w2id = new_w2id
        self.max_seq_len = max_len
        self.sents = [[new_w2id[word] for word in sent] for sent in new_sents]


    def line_to_embedding(self, line_num, vec_size, embeddings_file_lines):
        if line_num == -1:
            return torch.zeros(vec_size)

        line = embeddings_file_lines[line_num]
        all_tokens = line.split(" ")
        # vector = list(map(float, all_tokens[1:]))
        return torch.tensor(list(map(float, (all_tokens[1:]))))


    def read_embeddings(self, file_path):
        """Assumes that the first line of the file is
        the vocabulary length and vector dimension."""
        with open(file_path, encoding="utf8") as f:
            embeddings_file_lines = f.readlines()
        # vocab_length = len(txt)
        header = embeddings_file_lines[0]
        vec_size = int(header.split(" ")[1])
        embed_dict = {}
        w2id = {}
        w2id["<PAD>"] = 0
        embed_dict[0] = -1
        words_count = 1

        embeddings_file_lines = embeddings_file_lines[1:]
        for line_num, line in enumerate(embeddings_file_lines):
            all_tokens = line.split(" ")
            word = all_tokens[0]

            if word not in w2id:
                w2id[word] = words_count
                words_count += 1


            # vector = list(map(float, all_tokens[1:]))
            # vector = torch.tensor(list(map(float, (all_tokens[1:]))))
            word_id = w2id[word]
            embed_dict[word_id] = line_num

        w2id["<UNK>"] = words_count
        embed_dict[words_count] = -1
        id2w = dict(zip(w2id.values(), w2id.keys()))

        return w2id, id2w, embed_dict, vec_size, embeddings_file_lines
