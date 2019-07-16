import numpy as np

from data_processing.dataset_non_contextual import DatasetNonContextual
from data_processing.text_processors.text_processor_non_contextual import TextProcessorNonContextual


def main():
    x=3
    text_processor = TextProcessorNonContextual("../data/APRC/APRC_new1.txt",
                                                "../data/embeddings/wiki-news-300d-1M.vec", test_size=0.1,
                                                sents_limit=None, rare_word_threshold=10)

    data = DatasetNonContextual(text_processor.sents, text_processor.w2id, text_processor.id2w,
                                text_processor.max_seq_len, text_processor.max_masked_size, to_cuda=False)

    print("w2id: ", text_processor.w2id)
    print()
    print("id2w: ", text_processor.id2w)
    print()

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


def check_embeddings(file_path):
    """Assumes that the first line of the file is
    the vocabulary length and vector dimension."""
    with open(file_path) as f:
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

        if "<UNK" == word.upper()[:4]:
            print(word)
        # if word not in w2id:
        #     w2id[word] = words_count
        #     words_count += 1
        #
        # # vector = list(map(float, all_tokens[1:]))
        # vector = np.array(all_tokens[1:], dtype=float)
        # word_id = w2id[word]
        # emb_matrix[word_id] = vector

    # w2id["<UNK>"] = words_count
    # # emb_matrix[words_count] = vector
    id2w = dict(zip(w2id.values(), w2id.keys()))

    return vec_dim, w2id, id2w, emb_matrix


if __name__ == "__main__":
    main()
    # check_embeddings("../data/embeddings/wiki-news-300d-1M.vec")
