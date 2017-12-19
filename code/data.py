import csv 
import re
import numpy as np
import gensim
from nltk.corpus import conll2002
import nltk
import math
import sys 
from keras.preprocessing import text, sequence

class DataLoader(object):
    """docstring for DataLoader"""
    def __init__(self):
        super(DataLoader, self).__init__()
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.X_dev = None
        self.Y_dev = None
        self.tag_length = None
        self.w2v_size = None
        self.max_length =None
        self.input_path = None

    def get_file_data(self, path, embedding_path):
        print("loading data")
        nltk = False
        file = False
        self.input_path = path
        if path.endswith(".csv"):
            data = self.process_csv(path)
        elif path.endswith(".txt"):
            data = self.process_file(path)
            file = True
        else:
            # try:
            train, dev, test = self.load_conll(path.strip(), 30)
            print train[0]
            data = [[(x[0], x[2]) for x in sent ] for sent in train+dev+test]
            nltk = True
            # except:
                # print "invalid language option"
                # sys.exit(1)

        # print("getting X Y sets")
        X, Y, w2v_mapping, tag_embeddings, max_length = self.make_x_y(data, embedding_path, file, nltk)
        try:
            self.w2v_size = len(w2v_mapping[list(w2v_mapping.vocab)[0]])
            self.use_embedding_layer = False
        except AttributeError:
            self.w2v_size = 300
            self.use_embedding_layer = True
        train, test, dev = .7, .2, .1

        train_split = int(len(X)*train)
        test_split = train_split+1+int(math.floor(len(X)*test))

        self.X_train, self.Y_train = X[:train_split], Y[:train_split]
        self.X_test, self.Y_test = X[train_split +1 : test_split], Y[train_split + 1:test_split]
        self.X_dev, self.Y_dev = X[test_split +1 :], Y[test_split +1 :]

        print("X shape: train: {}, test: {}, dev: {}".format(self.X_train.shape, self.X_test.shape, self.X_dev.shape))
        print("Y shape: train: {}, test: {}, dev: {}".format(self.Y_train.shape, self.Y_test.shape, self.Y_dev.shape))

        self.tag_length = len(tag_embeddings[0].keys())

    def load_conll(self, version, max_length):
        train=conll2002.iob_sents(version+'.train')
        dev=[x for x in conll2002.iob_sents(version+'.testa') if len(x) <= max_length]
        test=[x for x in conll2002.iob_sents(version+'.testb') if len(x) <= max_length]

        return train, dev, test

    def process_csv(self, path):
        with open(path) as f1:
            reader = csv.DictReader(f1)
            sent = None
            to_ret = []
            sentence = []
            tag_sequence = []
            # add sentence info
            for i, row in enumerate(reader):
                if row["Sentence #"] is not "":
                    to_ret.append((sentence, tag_sequence))
                    sentence, tag_sequence = [], []
                else:
                    sentence.append(row["Word"])
                    tag_sequence.append(row["Tag"])

            return to_ret[1:]

    def process_file(self, path):
        with open(path) as f1:
            file = f1.read()
        line_sentences = re.split("(?<![\wO])\n", file)
        line_sentences = [re.split("\n", x) for x in line_sentences]
        line_sentences = [[re.split("\s", word_pair) for word_pair in sent ] for sent in line_sentences]

        return line_sentences

    def get_all_sentences(self, data, file=False):
        all_sents = []
        sentence = []
        if not file:
            all_sents = [zip(sent[0], sent[1]) for sent in data]

        if file:
            all_sents = [[pair for pair in sent if len(pair)>1 ] for sent in data]
        self.all_sents = all_sents
        return all_sents

    def get_nltk_sentences(self, data):
        """
        get all sentences from NLTK conll2002 corpus (in word-tag tuples)
        """
        all_sents = []
        for sentence in data:
            new_sent = []
            for tup in sentence: 
                new_tup = (tup[0], tup[1])
                new_sent.append(new_tup)
            all_sents.append(new_sent)   
        return all_sents

    def make_x_y(self, data, w2v_path=None, file=True, nltk=False):
        # check if .csv or .txt
        max_length = 30
        no_embedding = False
        if not nltk:
            all_sentences = self.get_all_sentences(data, file)
        else:
            all_sentences = self.get_nltk_sentences(data)


        all_sent_len = [len(x) for x in all_sentences]
        just_sents = [[y[0] for y in x] for x in all_sentences]
        max_len = max(all_sent_len)

        average_len = float(sum(all_sent_len))/float(len(all_sent_len))
        plot = False
        if plot:
            sent_len_dict = {x:0 for x in all_sent_len}
            for length in all_sent_len:
                sent_len_dict[length]+=1

            keys, values = zip(*sorted(sent_len_dict.items(), key=lambda x: x[0]))
            import matplotlib.pyplot as plt 
            plt.bar(keys, values)
            plt.show()
            print "this many sentences < 40"
            print float(sum([y for x,y in sent_len_dict.items() if x <= max_length]))/float(sum(values))


        original_num_sents = len(all_sentences)
        print "average sentence length: {}".format(average_len)
        print "omitting sentences longer than {}".format(max_length)
        all_sentences = [x for x in all_sentences if len(x) <= max_length]
        print "{} sentences out of {} were omitted ({}%)".format(original_num_sents - len(all_sentences), 
            original_num_sents, float(original_num_sents - len(all_sentences))/float(original_num_sents))
        if w2v_path is not None:
            print "loading w2v model"
            try:
                w2v_mapping = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
            except:
                w2v_mapping = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=False)
            
            size = len(w2v_mapping[list(w2v_mapping.vocab)[0]])
            w2v_vocab = w2v_mapping.vocab
            # get all words 
            all_words = set([tup[0] for sent in all_sentences for tup in sent])
            # get oov 
            oovs = all_words - set(w2v_vocab.keys())
            # # extend w2v by adding random vectors for oovs
            oov_dict = {}
            for oov in oovs:
                oov_vec = np.random.normal(0,1, size)
                oov_dict[oov] = oov_vec/np.sum(oov_vec)

        else:
            # if word2vec not used, need to assign each word a separate integer for embedding layer
            vocabulary = set([x for sent in just_sents for x in sent])
            self.vocab_size = len(vocabulary)
            size = 300
            encoded_sents = [text.one_hot(" ".join(sent), self.vocab_size) for sent in just_sents]
            # max_length = max([len(sent) for sent in just_sents])
            max_length = 30
            max_len = 30
            x_full = sequence.pad_sequences(encoded_sents, maxlen=max_length, padding='pre', truncating='post')
            w2v_mapping = None
            no_embedding = True
        # if file:
        all_tags = sorted(list(set([pair[1] for sent in all_sentences for pair in sent])))
        all_tags.append("NULL")
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
        if not no_embedding:
            x_full = []
        y_full = []

        for sent in all_sentences:
            w2v_seq = []
            tag_seq = []
            for i, tup in enumerate(sent):
                word, tag = tup
                if not no_embedding:
                    try:
                        w2v_seq.append(w2v_mapping[word])
                    except KeyError:
                        w2v_seq.append(oov_dict[word])
                # truncate tags
                if i < max_len:
                    tag_seq.append(tag_to_one_hot[tag])

            len_from_max = max_len - len(tag_seq)
            # was len(w2v_seq)
            if not no_embedding:
                x_full.append(np.array(((len_from_max)*[null_embedding]) + w2v_seq ))
            y_full.append(np.array(((len_from_max)*[null_tag]) + tag_seq))

        first_len = len(y_full[1])
        print "first_len"
        print first_len
        for y in y_full:
            try:
                assert(len(y) == first_len)
            except:
                print(len(y))
                print(y)
                sys.exit()

        x_full = np.array(x_full)
        y_full = np.array(y_full)
        print x_full.shape
        print y_full.shape
        self.one_hot_to_tag = one_hot_to_tag
        self.tag_to_one_hot = tag_to_one_hot
        return x_full, y_full, w2v_mapping, (tag_to_one_hot, one_hot_to_tag), max_len


