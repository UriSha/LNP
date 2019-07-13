import re
from collections import defaultdict

import numpy as np

from abstract_text_processor import AbstractTextProcessor


class TextProcessorNonContextual(AbstractTextProcessor):
    def __init__(self, text_file_path, embed_file_path, test_size=0.1, mask_ratio=.25, rare_word_threshold=10,
                 sents_limit=None):
        super(TextProcessorNonContextual, self).__init__(text_file_path, test_size, mask_ratio, rare_word_threshold,
                                                         sents_limit, embed_file_path=embed_file_path)
        print()
        print("init TextProcessorNonContextual")

    def normalize_word(self, w):
        w = w.replace("`", "")
        w = w.replace("'", "")
        lst = re.findall(r'[A-Za-z]+|\.', w)
        if len(lst) == 3:
            if lst[1] == '.':
                if len(lst[2]) == 1:
                    if lst[2] not in ['i', 'a']:
                        return lst[:2]
        return lst

    def initiate_vocab(self, sents):
        max_len = 0
        new_sents = []
        w2cnt = defaultdict(int)

        self.vec_size, w2id, id2w, self.embed_matrix = self._read_embeddings(self.embed_file_path, self.sents_limit)

        for sent in sents:
            new_sent = []
            for w in sent:
                w_list = self.normalize_word(w)
                if type(w_list) != list:
                    w_list = [w_list]

                for word in w_list:
                    if word not in w2id:  # i.e word has not embedding vector
                        word = "<UNK>"

                    new_sent.append(word)
                    w2cnt[word] += 1

            new_sents.append(new_sent)
            max_len = max(max_len, len(sent))

        # get rid of rare words in our sentences
        rare_words_count = 0
        for k in w2cnt.keys():
            if w2cnt[k] < self.rare_word_threshold and k != "<UNK>":
                rare_words_count += 1
                word_id = w2id[k]
                try:
                    del w2id[k]
                    del id2w[word_id]
                except KeyError:
                    print("k = ", k)
                    print("word_id = ", word_id)
        for sent in new_sents:
            for i in range(len(sent)):
                word = sent[i]
                if word not in w2id:
                    sent[i] = "<UNK>"
                    #  del self.embed_matrix[word_id]

        print(
            'With rare_word_threshold = {rare_word_threshold}, the ratio of rare words (that were removed) is: {ratio}'.format(
                rare_word_threshold=self.rare_word_threshold, ratio=rare_words_count / len(w2cnt)))

        w2id['<UNK>'] = len(w2id)
        return new_sents, w2id, id2w, max_len

    def _read_embeddings(self, file_path, sents_limit=None):
        """Assumes that the first line of the file is
        the vocabulary length and vector dimension."""
        with open(file_path) as f:
            if sents_limit:
                embeddings_file_lines = [next(f) for _ in range(sents_limit)]
            else:
                embeddings_file_lines = f.readlines()
        # vocab_length = len(txt)
        header = embeddings_file_lines[0]
        vec_dim = int(header.split(" ")[1])
        emb_matrix = {}
        w2id = {}
        w2id["<PAD>"] = 0

        words_count = 1

        embeddings_file_lines = embeddings_file_lines[1:]
        for line in embeddings_file_lines:
            all_tokens = line.split(" ")
            word = all_tokens[0]

            if word not in w2id:
                w2id[word] = words_count
                words_count += 1

            # vector = list(map(float, all_tokens[1:]))
            vector = np.array(all_tokens[1:], dtype=float)
            word_id = w2id[word]
            emb_matrix[word_id] = vector

        w2id["<UNK>"] = words_count
        id2w = dict(zip(w2id.values(), w2id.keys()))

        return vec_dim, w2id, id2w, emb_matrix
