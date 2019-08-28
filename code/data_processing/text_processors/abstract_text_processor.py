import math
import random


class AbstractTextProcessor:
    def __init__(self, text_file_path, test_size, rare_word_threshold, sents_limit, embed_file_path=None,
                 tokenizer=None):
        sents = self.read_data(text_file_path, sents_limit)
        self.sents_limit = sents_limit
        self.rare_word_threshold = rare_word_threshold
        self.embed_file_path = embed_file_path
        self.tokenizer = tokenizer
        self.orig_sents, self.w2id, self.id2w, self.max_seq_len = self.initiate_vocab(sents)
        # self.sents = self.remove_rare_words(sents)

        self.sents = [[self.w2id[word] for word in sent] for sent in self.orig_sents]
        shuffled_sents = random.sample(self.sents, k=len(self.sents))
        if type(test_size) == float:
            test_size = int(len(self.sents) * test_size)
        self.train_sents = shuffled_sents[:-test_size]
        self.test_sents = shuffled_sents[-test_size:]
        self.bleu_sents = random.sample(self.test_sents, k=min(10000, len(self.test_sents)))

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
                    i+=1
                    text.append(sent)
                    sent = next(f, None)
            else:
                text = f.readlines()

        return [sent.rstrip("\n").split(" ") for sent in text]

    # def remove_rare_words(self, sents):
    #     sentences = []
    #     for sent in sents:
    #         sentences.append([self.w2id.get(w, self.w2id['<UNK>']) for w in sent])
    #     return sentences

    def initiate_vocab(self, sents):
        raise Exception()
