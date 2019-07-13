from data_processing.dataset_consistent import DatasetConsistent
from data_processing.dataset_non_contextual import DatasetNonContextual
from data_processing.text_processors.text_processor_non_contextual import TextProcessorNonContextual
from model.cnp import CNP
from training import Trainer


def main():
    text_processor = TextProcessorNonContextual("../data/APRC/APRC_new1.txt",
                                                "../data/embeddings/wiki-news-300d-1M.vec", test_size=0.1,
                                                sents_limit=None, rare_word_threshold=10)

    data = DatasetNonContextual(text_processor.sents, text_processor.w2id, text_processor.id2w,
                                text_processor.max_seq_len, text_processor.max_masked_size, to_cuda=False)

    sent_ids_tensor, anti_mask_indices, paddings_mask, target_xs, target_ys = data[0]

    print("sent_ids_tensor: ", sent_ids_tensor)
    print("anti_mask_indices: ", anti_mask_indices)
    print("paddings_mask: ", paddings_mask)
    print("target_xs: ", target_xs)
    print("target_ys: ", target_ys)
    print()
    print()

    sent_ids_tensor, anti_mask_indices, paddings_mask, target_xs, target_ys = data[10]

    print("sent_ids_tensor: ", sent_ids_tensor)
    print("anti_mask_indices: ", anti_mask_indices)
    print("paddings_mask: ", paddings_mask)
    print("target_xs: ", target_xs)
    print("target_ys: ", target_ys)
    print()
    print()

    sent_ids_tensor, anti_mask_indices, paddings_mask, target_xs, target_ys = data[17]

    print("sent_ids_tensor: ", sent_ids_tensor)
    print("anti_mask_indices: ", anti_mask_indices)
    print("paddings_mask: ", paddings_mask)
    print("target_xs: ", target_xs)
    print("target_ys: ", target_ys)

    # text_processor = TextProcessor("data/APRC/APRC_small_mock.txt", test_size=0.05, sents_limit=5)


if __name__ == "__main__":
    main()
