from collections import defaultdict

from pytorch_pretrained_bert import BertTokenizer

from .abstract_text_processor import AbstractTextProcessor


class TextProcessor(AbstractTextProcessor):
    def __init__(self, text_file_path, test_size=0.1, mask_ratio=.25, rare_word_threshold=10, sents_limit=None):
        super(TextProcessor, self).__init__(text_file_path, test_size, mask_ratio,
                                            rare_word_threshold, sents_limit, tokenizer=BertTokenizer.from_pretrained('bert-base-uncased'))
        print()
        print("init TextProcessor")

    def get_sent_long_version(self, sent):
        sent_as_string = " ".join(sent)
        tokenized_sent = self.tokenizer.tokenize(sent_as_string)
        return tokenized_sent

    def initiate_vocab(self, sents):
        max_len = 0
        w2id = {}
        w2id['<PAD>'] = 0
        id2w = {}

        w2cnt = defaultdict(int)

        for sent in sents:
            for w in sent:
                w2cnt[w] += 1

            sent = self.get_sent_long_version(sent)
            max_len = max(max_len, len(sent))

        words_count = 1
        rare_words_count = 0
        for k in w2cnt.keys():
            if w2cnt[k] < self.rare_word_threshold:
                rare_words_count += 1
            w2id[k] = words_count
            id2w[words_count] = k
            words_count += 1

        print(
            'With rare_word_threshold = {rare_word_threshold}, the ratio of rare words is: {ratio}'.format(
                rare_word_threshold=self.rare_word_threshold, ratio=rare_words_count / len(w2cnt)))

        w2id['<UNK>'] = len(w2id)
        return sents, w2id, id2w, max_len
