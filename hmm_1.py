from nltk.corpus import conll2002
import nltk
from nltk.tag import hmm

#train data- pre-tagged

def data_loader(version):
	train=conll2002.sents(version+'.train')
	dev=conll2002.sents(version+'.testa')
	test=conll2002.sents(version+'.testb')
	return train, dev, test

train_esp, dev_esp, test_esp=data_loader('esp')
