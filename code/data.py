import csv 
import re
import numpy as np
import gensim
from nltk.corpus import conll2002, stopwords
import nltk
import math
import sys 
from keras.preprocessing import text, sequence

class DataLoader(object):
    """DataLoader class creates passable object containing all data"""
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

    def get_file_data(self, path, embedding_path, stopwords=None):
        """
        for LSTM loading, read in the data from a path (or nltk language option) and transform this into 
        a set of questions (X) and answers (Y). If an embedding is used, then the X array will be a 3d numpy array 
        where the 1st dimension is the number of sentences, the 2nd is the length of each sentence, and the 3rd is 
        the embedding for each word in the sentence. If no embedding is used, X will be a 2d array where the 1st 
        dimension is the number of sentence and the 2nd is the length of the sentence. Each entry in X corresponds to
        a sentence, and each entry in the sentence is a word, either in w2v vector form or with each word type mapped
        to a discrete integer for the embedding layer.
        
        Tag types are counted and each tag is similarly transformed into a one-hot vector encoding its tag type. This is
        the Y array, composed of a list of sentences, where each sentence is a sequence of one-hot tag arrays. 

        This method also splits the data into the train,test, and dev sets.

        Parameters
        ----------
        path: str
            path to the data (csv, txt, or nltk language option ("ned" or "esp"))
        embedding_path: str
            path to the embedding file (binary or text) if available
        stopwords: str
            nltk option for stopwords, if set to valid string stopwords are removed 
        """
        print("loading data")
        use_nltk = False
        file = False
        self.input_path = path
        if path.endswith(".csv"):
            data = self.process_csv(path)
        elif path.endswith(".txt"):
            data = self.process_file(path)
            file = True
        else:
            train, dev, test = self.load_conll(path.strip())
            data = [[(x[0], x[2]) for x in sent ] for sent in train+dev+test]
            use_nltk = True

        print("getting X Y sets")
        # print data[100]
        X, Y, w2v_mapping, tag_embeddings, max_length = self.make_x_y(data, embedding_path, file, use_nltk, stopwords)
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
        sys.exit()
    def load_conll(self, version):
        """
        loads conll 2002 datasets from nltk

        Parameters
        ----------
        version: str
            either ned or esp

        Returns
        -------
        tuple
            train, test, and dev sets
        """
        train=conll2002.iob_sents(version+'.train')
        dev=conll2002.iob_sents(version+'.testa')
        test=conll2002.iob_sents(version+'.testb')

        return train, dev, test

    def process_csv(self, path):
        """
        loads sentences from csv format

        Parameters
        ----------
        path: str
            path to the csv

        Returns
        -------
        to_ret: list
            a list of tuples of the form (word_sequence, tag_sequence)
        """
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
        """
        takes a list of tuples of the form (word_sequence, tag_sequence)
        transforms into a list of lists of tuples of the form [[[#sent1(word1, tag1),(word2, tag2)... ], #sent2...]]

        Parameters
        ----------
        data: list
            list of tuples of the form (word_sequence, tag_sequence)

        Returns
        -------
        all_sents: list
            list of lists of tuples of the form [[[#sent1(word1, tag1),(word2, tag2)... ], #sent2...]]
        """
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

    def get_most_common(self, path):
        """
        gets top two most common tags in a corpus 

        Parameters
        ----------
        path: str
            path to the corpus

        Returns
        -------
        sentences: list
            a list of sentences (for length later)
        top: tuple
            the top two tags
        """
        print("loading data")
        if path.endswith(".csv"):
            data = self.process_csv(path)
            sentences = self.get_all_sentences(data)
        elif path.endswith(".txt"):
            data = self.process_file(path)
            file = True
            sentences = self.get_all_sentences(data)
        else:
            # try:
            train, dev, test = self.load_conll(path.strip(), 30)
            print train[0]
            data = [[(x[0], x[2]) for x in sent ] for sent in train+dev+test]
            sentences = data
        
        just_tags = [y[1] for x in sentences for y in x ]
        tagset = set(just_tags)
        tag_freq = {x:0 for x in tagset}
        for t in just_tags:
            tag_freq[t] +=1

        print "tag frequencies:"
        print tag_freq
        print "top tags"
        top =  sorted(tag_freq.items(), key = lambda x: x[1])
        return sentences, top[-2:]



    def make_x_y(self, data, w2v_path=None, file=True, use_nltk=False, stopwords=None):
        """ 
        transforms data where each sentence is a list of tuples (word, tag) into embedded vectors (either discrete 
        or word2vec) and tag embeddings.

        Truncates sentences to be below the 90th percentile w.r.t. length for scalability reasons
        pads shorter sentences with null embedding. 

        Parameters
        ----------
        data: list
            list of sentences where each sentence is a list of tuples (word, tag)
        w2v_path: str
            path to pretrained w2v embeddings. Defaults to None
        file: boolean
            whether data is read from .txt or .csv. Defaults to True
        use_nltk: boolean
            whether data is to be read from nltk corpus. Defaults to False
        stopwords: str
            which stopword language option to use. Defaults to None

        Returns
        -------
        x_full: np.array
            array representation of all sentences (w2v or discrete)
        y_full: 
            array representation of all corresponding tag sequences (one-hot embedding)
        w2v_mapping:
            the mapping of words to embedded vectors 
        tag_to_one_hot:
            the mapping of tags to one-hot vectors
        one_hot_to_tag:
            the mapping of one-hot vectors to tags (needed for decoding)
        max_len:
            the maximum length of sentences 
        """

        # check if .csv or .txt
        
        no_embedding = False
        if not use_nltk:
            all_sentences = self.get_all_sentences(data, file)
        else:
            all_sentences = self.get_nltk_sentences(data)

        if stopwords is not None:
            stop = set(nltk.corpus.stopwords.words(stopwords)) | set([".",",","(",")",";","?","!",":"])
            all_sentences = [[tup for tup in sent if tup[0] not in stop] for sent in all_sentences]

        print(all_sentences[0])
        all_sent_len = [len(x) for x in all_sentences]

        average_len = float(sum(all_sent_len))/float(len(all_sent_len))
        sent_len_dict = {x:0 for x in all_sent_len}
        for length in all_sent_len:
            sent_len_dict[length]+=1

        # compute cutoff 
        # whatever length keeps 90% of sentences
        ninty_break = .90*len(all_sentences)
        length_sum = 0
        sentence_index = 0
        for length, sent_num in sorted(sent_len_dict.items(), key=lambda x: x[0]):
            sentence_index = length
            length_sum += sent_num
            if length_sum > ninty_break:
                break
        max_length = sentence_index
        max_len = max_length

        # plot if needed
        plot = False
        if plot:
            keys, values = zip(*sorted(sent_len_dict.items(), key=lambda x: x[0]))
            import matplotlib.pyplot as plt 
            plt.bar(keys, values)
            plt.axvline(x=max_length, **{"color":"red"})

            plt.show()
            print "this many sentences < {}".format(max_length)
            print float(sum([y for x,y in sent_len_dict.items() if x <= max_length]))/float(sum(values))

        original_num_sents = len(all_sentences)
        print "average sentence length: {}".format(average_len)
        print "omitting sentences longer than {}".format(max_length)
        all_sentences = [x for x in all_sentences if len(x) <= max_length]
        print "max length is now: {}".format(max([len(x) for x in all_sentences]))
        print "{} sentences out of {} were omitted ({}%)".format(original_num_sents - len(all_sentences), 
            original_num_sents, float(original_num_sents - len(all_sentences))/float(original_num_sents))
        
        just_sents = [[y[0] for y in x] for x in all_sentences]
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
            if not no_embedding:
                x_full.append(np.array(((len_from_max)*[null_embedding]) + w2v_seq ))
            y_full.append(np.array(((len_from_max)*[null_tag]) + tag_seq))

        first_len = len(y_full[1])
        # integrity check
        print "first_len: {}".format(first_len)
        for y in y_full:
            try:
                assert(len(y) == first_len)
            except:
                print(len(y))
                print(y)
                sys.exit()

        print "asserted"
        print "number of sentences: {}".format(len(x_full))
        print "length of sentence: {}".format(len(x_full[0]))

        x_full = np.array(x_full)
        y_full = np.array(y_full)
        print x_full.shape
        print y_full.shape
        self.one_hot_to_tag = one_hot_to_tag
        self.tag_to_one_hot = tag_to_one_hot
        return x_full, y_full, w2v_mapping, (tag_to_one_hot, one_hot_to_tag), max_len


