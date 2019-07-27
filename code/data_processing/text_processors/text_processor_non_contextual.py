import re
from collections import defaultdict

import torch

from .abstract_text_processor import AbstractTextProcessor


class TextProcessorNonContextual(AbstractTextProcessor):
    def __init__(self, text_file_path, embed_file_path, test_size=0.1, mask_ratio=.25, rare_word_threshold=10,
                 sents_limit=None):
        super(TextProcessorNonContextual, self).__init__(text_file_path, test_size, mask_ratio, rare_word_threshold,
                                                         sents_limit, embed_file_path=embed_file_path)

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

    def initiate_vocab(self, sents):
        max_len = 0
        new_sents = []
        w2cnt = defaultdict(int)

        temp_w2id, temp_id2w, embed_dict = self._read_embeddings(self.embed_file_path)

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
            if w2cnt[k] < self.rare_word_threshold and k != "<UNK>" and k != "<PAD>":
                rare_words_count += 1
                word_id = temp_w2id[k]
                try:
                    del temp_w2id[k]
                    del temp_id2w[word_id]
                    del embed_dict[word_id]
                except KeyError:
                    print("k = ", k)
                    print("word_id = ", word_id)

        
        embed_list = [torch.zeros(self.vec_size)]
        new_w2id = {}
        new_w2id["<PAD>"] = 0
        new_id2w = {}
        new_id2w[0] = "<PAD>"
        id_freq = defaultdict(int)
        for sent in new_sents:
            for i in range(len(sent)):
                word = sent[i]
                if word not in temp_w2id:
                    sent[i] = "<UNK>"
                elif word not in new_w2id:
                    old_word_id = temp_w2id[word]
                    new_word_id = len(embed_list)
                    embed_list.append(self._line_to_embedding(embed_dict[old_word_id]))
                    new_w2id[word] = new_word_id
                    new_id2w[new_word_id] = word
                id_freq[new_w2id[word]] += 1

        if "<UNK>" not in new_w2id:
            new_w2id["<UNK>"] = len(embed_list)
            new_id2w[len(embed_list)] = "<UNK>"
            embed_list.append(torch.zeros(self.vec_size))
            id_freq[new_w2id["<UNK>"]] += 1

        # rearrange dicts by frequency
        sorted_id2w = {}
        sorted_id2w[0] = "<PAD>"
        sorted_w2id = {}
        sorted_w2id["<PAD>"] = 0
        sorted_embed_list = [torch.zeros(self.vec_size)]
        s = [(k, id_freq[k]) for k in sorted(id_freq, key=id_freq.get, reverse=True)]
        for i, (k, v) in enumerate(s):
            word_id = i+1
            word = new_id2w[k]
            sorted_id2w[word_id] = word
            sorted_w2id[word] = word_id
            sorted_embed_list.append(embed_list[k])

        embed_list = sorted_embed_list
        new_id2w = sorted_id2w
        new_w2id = sorted_w2id

        print(
            'With rare_word_threshold = {rare_word_threshold}, the ratio of rare words (that were removed) is: {ratio}'.format(
                rare_word_threshold=self.rare_word_threshold, ratio=rare_words_count / len(w2cnt)))

        
        # for idx, (old_word_id, word) in enumerate(temp_id2w.items()):
        #     embed_vector = embed_dict[old_word_id]

        #     embed_list.append(embed_vector)
        #     new_w2id[word] = idx
        #     new_id2w[idx] = word

        # with open('/Users/omerkoren/Final_Project/TICNP/data/embeddings/APRC_embeddings.txt', 'w+') as f:
        #     f.write('999994 300\n')
        #     for idx, embed_vector in enumerate(embed_list):
        #         word = new_id2w[idx]
        #         f.write('{word} {vec}\n'.format(word=word, vec=' '.join(embed_vector)))


        self.embeddings_file_lines = None
        self.embed_matrix = torch.stack(embed_list)

        return new_sents, new_w2id, new_id2w, max_len

    def _line_to_embedding(self, line_num):
        if line_num == -1:
            return torch.zeros(self.vec_size)
        
        line = self.embeddings_file_lines[line_num]
        all_tokens = line.split(" ")
        # vector = list(map(float, all_tokens[1:]))
        return torch.tensor(list(map(float, (all_tokens[1:]))))

    def _read_embeddings(self, file_path):
        """Assumes that the first line of the file is
        the vocabulary length and vector dimension."""
        with open(file_path, encoding="utf8") as f:
            embeddings_file_lines = f.readlines()
        # vocab_length = len(txt)
        header = embeddings_file_lines[0]
        self.vec_size = int(header.split(" ")[1])
        emb_matrix = {}
        w2id = {}
        self.pad_index = 0
        w2id["<PAD>"] = self.pad_index
        emb_matrix[self.pad_index] = -1
        words_count = 1

        self.embeddings_file_lines = embeddings_file_lines[1:]
        for line_num, line in enumerate(self.embeddings_file_lines):
            all_tokens = line.split(" ")
            word = all_tokens[0]

            if word not in w2id:
                w2id[word] = words_count
                words_count += 1


            # vector = list(map(float, all_tokens[1:]))
            # vector = torch.tensor(list(map(float, (all_tokens[1:]))))
            word_id = w2id[word]
            emb_matrix[word_id] = line_num

        w2id["<UNK>"] = words_count
        emb_matrix[words_count] = -1
        id2w = dict(zip(w2id.values(), w2id.keys()))

        return w2id, id2w, emb_matrix
