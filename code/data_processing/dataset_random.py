from pytorch_pretrained_bert import BertModel

from data_processing.abstract_dataset import AbstractDataset


class DatasetRandom(AbstractDataset):
    def __init__(self, text_as_list, tokenizer, w2id, max_seq_len, max_masked_size, mask_ratio=.25, transform=None,
                 to_cuda=True):
        super(DatasetRandom, self).__init__(text_as_list=text_as_list,
                                            tokenizer=tokenizer,
                                            w2id=w2id,
                                            max_seq_len=max_seq_len,
                                            max_masked_size=max_masked_size,
                                            mask_ratio=mask_ratio,
                                            transform=transform,
                                            to_cuda=to_cuda)

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = tokenizer
        self.bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        self.bert_pretrained.eval()

        if self.to_cuda:
            self.bert_pretrained.to('cuda')

    def __getitem__(self, index):
        return self.generate_data_instance_fron_sentence(original_sent=self.data[index],
                                                         tokenizer=self.tokenizer,
                                                         bert_pretrained=self.bert_pretrained)
