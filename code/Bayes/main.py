from bayes import *
from utils import *

if __name__ == "__main__":
    docs_context, docs_labels = create_data()
    vocabs = create_vocab_list(docs_context)
    train_mat = []
    for doc in docs_context:
        train_mat.append(set_of_words_2_vec(vocabs, doc))
    p_vec, pAbusive = cond_probability(train_mat, docs_labels)
    # print(p_vec)
    # print(pAbusive)
    # exit()
    test_entry = ['love', 'my', 'dalmation']
    thisDoc = set_of_words_2_vec(vocabs, test_entry)
    print(test_entry, 'classified as: ', classifyNB(thisDoc, p_vec, pAbusive))

    test_entry = ['stupid', 'garbage']
    thisDoc = set_of_words_2_vec(vocabs, test_entry)
    print(test_entry, 'classified as: ', classifyNB(thisDoc, p_vec, pAbusive))
