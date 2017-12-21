import re
import argparse
import sys
import csv
from nltk.tag import hmm

from base_lstm import define_model
from data import DataLoader
from tester import evaluate_f1
from tester import evaluate_hmm
from nltk.probability import *
from tester import hmm_sequences

# embedding_size, tag_size, input_length, embedding = False, vocab_dim = 0
def run_lstm(loader, output_file, use_dev, layer_num, resume_path):
    print "defining model..."
    if loader.use_embedding_layer:
        model = define_model(loader.w2v_size, loader.tag_length, loader.max_length, 
            layer_num, True, loader.vocab_size)
    else:
        model = define_model(loader.w2v_size, loader.tag_length, loader.max_length, layer_num)

    print "model summary:"
    print model.summary()

    print "training..." 
    if resume_path is not None:
        model.load_weights(resume_path)
    model.fit(loader.X_train, loader.Y_train, epochs=20, batch_size=32, verbose=1,  shuffle=True)
    model.save("../models/{}-{}-{}".format(loader.input_path, loader.use_embedding_layer, loader.w2v_size))
    print "evaluating"
    if use_dev:
        results = model.evaluate(loader.X_dev, loader.Y_dev, verbose=1)
        f1_results = evaluate_f1(model, loader, use_dev)
    else:
        results = model.evaluate(loader.X_test, loader.Y_test, verbose=1)
        f1_results = evaluate_f1(model, loader, use_dev)

    with open(output_file, "a") as f1:
        writer = csv.writer(f1)
        # row format: pretrained_embedding?, embedding_size, number of layers, accuarcy, f1
        row = [loader.use_embedding_layer, loader.w2v_size, layer_num, results[1], f1_results]
        writer.writerow(row)
    return results, f1_results

def run_hmm(loader, use_dev, estimator, experiment):
    print "training..."
    trainer=hmm.HiddenMarkovModelTrainer()
    if estimator=='Laplace':
        if experiment=='1':
            tagger = trainer.train_supervised(loader.train, LaplaceProbDist)
        else:
            tagger = trainer.train_supervised(loader.test, LaplaceProbDist)
    else:
        if experiment=='1':
            tagger = trainer.train_supervised(loader.train)
        else:
            tagger = trainer.train_supervised(loader.test, LaplaceProbDist)
    print "evaluating..."
    accuracy, f1= evaluate_hmm(loader, tagger, use_dev, experiment)
    print ("The accuracy is: {} and the f1 is:{}".format(accuracy, f1))

if __name__ == '__main__':
    USE_DEV = False
    LAYER_NUM = 1
    # todo: vary number of epochs 

    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_file", dest='embedding_path', 
        default=None, required = False,
        help="optional path to an embedding file. If left blank, no pre-trained embeddings will be used")
    parser.add_argument("--split", dest='train_split',
        default=".7-.2-.1", help="train-test-dev split, default is .7, .2, .1")
    parser.add_argument("--input_file", dest="input_path", required=True,
        help="mandatory path to NER corpus, either in csv or txt format")
    parser.add_argument("--model_type", dest="model_type", required=True, type=str, 
        help="mandatory argument: what type of NER system to run. \n Expected: 'lstm' or 'hmm'")
    parser.add_argument("--resume_model", dest="resume_path", required=False, type=str, help="resume model from existing file")
    parser.add_argument("--estimator", dest="estimator", default="Laplace", required=False, type=str, help="if using an hmm, choose an estimator \n Expected: 'mle' or 'Laplce'")
    parser.add_argument("--experiment", dest="experiment", default="1", required=True, type=str, help="Choose experiment 1 or 2")
    args = parser.parse_args()

    loader = DataLoader()

    
    loader.get_file_data(args.input_path, args.embedding_path, args.model_type)
    
    if args.model_type.lower() == "lstm":
        run_lstm(loader, "../results/lstm_results_ned.csv", USE_DEV, LAYER_NUM, args.resume_path)
    else:
        run_hmm(loader, USE_DEV, args.estimator, args.experiment)
