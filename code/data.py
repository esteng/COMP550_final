import csv 
import re
import gensim
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

# from keras.preprocessing.sequence import pad_sequences


def process_csv(path):
    with open(path) as f1:
        reader = csv.DictReader(f1)
        sent = None
        to_ret = []
        # add sentence info
        for i, row in enumerate(reader):
            if row["Sentence #"] is not None:
                sent = row["Sentence #"]
            else:
                row["Sentence #"] = sent

            to_ret.append(row)
        return to_ret

def process_file(path):
    with open(path) as f1:
        file = f1.read()
    line_sentences = re.split("(?<![\wO])\n", file)
    print("there are {} sentences".format(len(line_sentences)))
    line_sentences = [re.split("\n", x) for x in line_sentences]
    line_sentences = [[re.split("\s", word_pair) for word_pair in sent ] for sent in line_sentences]

    return line_sentences



def get_all_sentences(data, file=False):
    all_sents = []
    sentence = []
    if not file:
        current_sent = data[0]["Sentence #"]
        for i, row in enumerate(data): 
            if current_sent == row["Sentence #"]:
                sentence.append((row["Word"], row["Tag"]))
            else:
                all_sents.append(sentence)
                sentence = [(row["Word"], row["Tag"])]
                current_sent = row["Sentence #"]

    if file:
        all_sents = [[pair for pair in sent if len(pair)>1 ] for sent in data]
    return all_sents

def make_x_y(data, size, file=False):
    all_sentences = get_all_sentences(data, file)

    #shuffle
    # np.random.shuffle(all_sentences)

    all_sent_len = [len(x) for x in all_sentences]
    just_sents = [[y[0] for y in x] for x in all_sentences]
    max_len = max(all_sent_len)
    print("Max sentence length is : {}".format(max_len))
    # w2v_mapping = gensim.models.Word2Vec(just_sents, min_count=1, size = size)
    # model.save("testmodel.wv")
    # sys.exit()
    print "loading w2v model"
    w2v_mapping = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    # model = gensim.models.KeyedVectors.load_word2vec_format("wordvecs.txt", binary=False)
    # wordvecs = pd.read_table("wordvecs.txt", sep='\t', header=None)
    # print(wordvecs)
    # print(wordvecs.drop(wordvecs.columns[[-1]], axis=1))
    # w2v_mapping = {}
    # with open("wordvecs.txt") as f1:
    #     lines = f1.readlines()
    #     for line in lines:
    #         splitline = line.strip().split("\t")
    #         word = splitline [0]
    #         vector = np.array([x for x in splitline[1:] if x is not None])
    #         w2v_mapping[word] = vector


    # w2v_vocab = w2v_mapping.keys()
    # w2v_vocab = w2v_mapping.vocab
    # get all words 
    all_words = set([tup[0] for sent in all_sentences for tup in sent])
    # get oov 
    # oovs = all_words - set(w2v_vocab.keys())
    # print "there are {} oovs".format(len(oovs))
    # # extend w2v by adding random vectors for oovs
    oov_dict = {}
    for oov in oovs:
        oov_vec = np.random.normal(0,1, 300)
        oov_dict[oov] = oov_vec/np.sum(oov_vec)

    if not file:
        all_tags = sorted(list(set([row["Tag"] for row in data])))
    if file:
        all_tags = sorted(list(set([pair[1] for sent in all_sentences for pair in sent])))

    all_tags.append("NULL")
    print("there are {} tags".format(len(all_tags)))
    tag_to_one_hot = {}
    one_hot_to_tag = {}
    for i, tag in enumerate(all_tags):
        tag_vector = np.zeros((len(all_tags)))
        tag_vector[i] = 1
        tag_to_one_hot[tag] = tag_vector
        one_hot_to_tag[tuple(tag_vector)] = tag

    null_embedding = np.zeros((size))
    null_tag = np.zeros((len(tag_vector)))
    null_tag[-1] = 1

    tag_to_one_hot["NULL"] = null_tag
    one_hot_to_tag[tuple(null_tag)]  = "NULL"
    print "there are this many tags:"

    x_full = []
    y_full = []

    for sent in all_sentences:
        w2v_seq = []
        tag_seq = []
        for i, tup in enumerate(sent):
            word, tag = tup
            try:
                # w2v_seq.append(model[word])
                w2v_seq.append(w2v_mapping[word])
            except KeyError:
                w2v_seq.append(oov_dict[word])
            tag_seq.append(tag_to_one_hot[tag])
      
        len_from_max = max_len - len(w2v_seq)


        x_full.append(np.array(((len_from_max)*[null_embedding]) + w2v_seq ))
        y_full.append(np.array(((len_from_max)*[null_tag]) + tag_seq))


    x_full = np.array(x_full)
    y_full = np.array(y_full)
    print x_full.shape
    print y_full.shape
    return x_full, y_full, w2v_mapping, (tag_to_one_hot, one_hot_to_tag)



# data = process_csv("../data/entity-annotated-corpus/ner_dataset.csv")
# X, Y, word_embeddings, tag_embeddings = make_x_y(data, 300)


