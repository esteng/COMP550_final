from nltk.corpus import conll2002
import nltk
from nltk.tag import hmm
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import Counter

#train data- pre-tagged

def data_loader(version):
	train=conll2002.raw(version+'.train')
	dev=conll2002.raw(version+'.testa')
	test=conll2002.raw(version+'.testb')
	return label(train.encode('ascii', 'ignore')), label(dev.encode('ascii', 'ignore')), label(test.encode('ascii', 'ignore'))

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
	tagger = trainer.train_supervised(train_esp)
	tagger.test(dev_esp)

def baseline(train, test):
	tags=[]
	for sent in train:
		tags.append([x[1] for x in sent])
	flat_tags=[y for sublist in tags for y in sublist]
	commonality=Counter(flat_tags)
	most_common=commonality.most_common(1)[0][0]
	print most_common
	correct=0
	total=0
	for sent in test:
		for word in sent:
			total+=1
			if most_common==word[1]:
				correct+=1
	return float(correct)/total


#tag('esp')
train_esp, dev_esp, test_esp=data_loader('esp')
print baseline(train_esp, dev_esp)