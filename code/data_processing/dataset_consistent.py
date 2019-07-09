from pytorch_pretrained_bert import BertModel

from code.data_processing.abstract_dataset import AbstractDataset


class DatasetConsistent(AbstractDataset):
    def __init__(self, text_as_list, tokenizer, w2id, max_seq_len, max_masked_size, mask_ratio=.25, transform=None,
                 to_cuda=True):
        super(DatasetConsistent, self).__init__(text_as_list=text_as_list,
                                                tokenizer=tokenizer,
                                                w2id=w2id,
                                                max_seq_len=max_seq_len,
                                                max_masked_size=max_masked_size,
                                                mask_ratio=mask_ratio,
                                                transform=transform,
                                                to_cuda=to_cuda)

        # Load pre-trained model tokenizer (vocabulary)
        tokenizer = self.get_tokenizer()

        self.data = self.create_data(text_as_list, tokenizer)

    def create_data(self, text_as_list, tokenizer):
        bert_pretrained = BertModel.from_pretrained('bert-base-uncased')
        bert_pretrained.eval()

        if self.to_cuda:
            bert_pretrained.to('cuda')

        data = []
        for original_sent in text_as_list:
            data_instance = self.generate_data_instance_fron_sentence(original_sent=original_sent,
                                                                      tokenizer=tokenizer,
                                                                      bert_pretrained=bert_pretrained)
            data.append(data_instance)

        return data

    def __getitem__(self, index):
        return self.data[index]
