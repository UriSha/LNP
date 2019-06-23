import pickle as cPickle

if __name__ == "__main__":
    print('Hello World!')
    name = 'APRC'

    # mid25 = cPickle.load(open('../data/%s/50_middle.pkl' % name, 'rb'))
    index = cPickle.load(open('../data/%s/index.pkl' % name, 'rb'))
    w2id, id2w = cPickle.load(open('../data/%s/w2id_id2w.pkl' % name, 'rb'))

    with open('../data/embeddings/wiki-news-300d-1M.vec', encoding='utf-8') as fin:
        lines_embeddings = fin.readlines()
        print(len(lines_embeddings))

    word_with_embeddings_vector = set([lines_embeddings[i].split(' ')[0] for i in range(1, len(lines_embeddings))])
    print(len(word_with_embeddings_vector))
    # print("'s" in word_with_embeddings_vector)
    # data_mid25 = [[id2w[id] for id in sent[0]] for sent in mid25]
    data_index = [[id2w[id] for id in sent] for sent in index]
    all_words = set()
    for sent in data_index:
        for w in sent:
            if w != "<EOS>":
                all_words.add(w)

    x = 4
