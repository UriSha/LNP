import math

from pytorch_pretrained_bert import BertTokenizer
from sklearn.model_selection import train_test_split


class TextProcessor:
    def __init__(self, text_file_path, sents_limit=0, test_size=0.1, mask_ratio=.25):
        print()
        print("init TextProcessor")
        sents = self.read_data(text_file_path)
        if sents_limit > 0:
            sents = sents[:sents_limit]
        self.sents = sents
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_sents, self.eval_sents = train_test_split(self.sents, test_size=test_size)
        self.w2id, self.id2w, self.max_seq_len = self.initiate_vocab()
        self.max_masked_size = math.ceil(self.max_seq_len * mask_ratio)

    def read_data(self, path):
        with open(path, "r") as f:
            text = f.readlines()
        return [sent.rstrip("\n").split(" ") for sent in text]

    def initiate_vocab(self):
        max_len = 0
        words_count = 0
        w2id = {}
        id2w = {}

        num_of_sents = len(self.sents)
        one_tenth = num_of_sents / 10
        cur_count = 0
        sents_processed = 0

        for sent in self.sents:
            for w in sent:
                if w not in w2id:
                    w2id[w] = words_count
                    id2w[words_count] = w
                    words_count += 1

            sent_as_string = " ".join(sent)
            tokenized_sent = self.tokenizer.tokenize(sent_as_string)
            max_len = max(max_len, len(tokenized_sent))
            sents_processed += 1
            cur_count += 1

            if cur_count == one_tenth:
                print("Processed {} out of {} sentences".format(sents_processed, num_of_sents))
                cur_count = 0

        return w2id, id2w, max_len
