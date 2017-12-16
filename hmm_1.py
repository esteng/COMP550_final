from nltk.corpus import conll2002
import nltk
from nltk.tag import hmm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

#train data- pre-tagged

def data_loader(version):
	train=conll2002.raw(version+'.train')
	dev=conll2002.raw(version+'.testa')
	test=conll2002.raw(version+'.testb')
	return train.encode('ascii', 'ignore'), dev.encode('ascii', 'ignore'), test.encode('ascii', 'ignore')

def label(data):
	line_tokenized=data.splitlines()
	word_tokenized_lines=[]
	for line in line_tokenized:
		word_tokenized_lines.append(word_tokenize(line))
	labelled=[]
	sent=[]
	for l in word_tokenized_lines:
		if len(l)>2:
			if l[0]=='.':
				if len(l)==3:
					sent.append((l[0], l[2]))
				else:
					sent.append((l[0], l[1]))
				labelled.append(sent)
				sent=[]
			else:
				if len(l)==3:
					sent.append((l[0], l[2]))
				else:
					sent.append((l[0], l[1]))
	return labelled

def tag(version):
	train_esp, dev_esp, test_esp=data_loader(version)
	trainer=hmm.HiddenMarkovModelTrainer()
	train_data=label(train_esp)
	tagger = trainer.train_supervised(train_data)
	dev_data=label(dev_esp)
	tagger.test(dev_data)

tag('esp')