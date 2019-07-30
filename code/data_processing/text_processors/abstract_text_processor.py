import math

from sklearn.model_selection import train_test_split


class AbstractTextProcessor:
    def __init__(self, text_file_path, test_size, mask_ratio, rare_word_threshold, sents_limit, use_weight_loss=True, embed_file_path=None,
                 tokenizer=None):
        sents = self.read_data(text_file_path, sents_limit)
        self.use_weight_loss = use_weight_loss
        self.sents_limit = sents_limit
        self.rare_word_threshold = rare_word_threshold
        self.embed_file_path = embed_file_path
        self.tokenizer = tokenizer
        self.sents, self.w2id, self.id2w, self.max_seq_len, self.word_weights = self.initiate_vocab(sents)
        # self.sents = self.remove_rare_words(sents)
        self.max_masked_size = math.ceil(self.max_seq_len * mask_ratio)
        self.train_sents, self.eval_sents = train_test_split(self.sents, test_size=test_size)

    def read_data(self, path, sents_limit):
        import os; print(os.getcwd())
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
