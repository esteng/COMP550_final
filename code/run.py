import re
import argparse
import sys

from base_lstm import define_model
from data import DataLoader
from tester import evaluate_f1



# embedding_size, tag_size, input_length, embedding = False, vocab_dim = 0
def run_lstm(loader, use_dev=False):
    print("defining model...")
    if loader.use_embedding_layer:
        model = define_model(loader.w2v_size, loader.tag_length, loader.max_length, True, loader.vocab_size)
    else:
        model = define_model(loader.w2v_size, loader.tag_length, loader.max_length)
    print "model summary:"
    print model.summary()
    print("training...")
    model.fit(loader.X_train, loader.Y_train, epochs=1, batch_size=32, verbose=1,  shuffle=True)
    print "evaluating"
    if use_dev:
        # results = model.evaluate(loader.X_dev, loader.Y_dev, verbose=1)
        f1_results = evaluate_f1(model, loader, True)
    else:
        # results = model.evaluate(loader.X_test, loader.Y_test, verbose=1)
        f1_results = evaluate_f1(model, loader, False)

    print "accuracy"
    print results

    print "f1 score:"
    print f1_results

    return results, f1_results

if __name__ == '__main__':
    USE_DEV = True

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
    args = parser.parse_args()

    loader = DataLoader()

    
    loader.get_file_data(args.input_path, args.embedding_path)
    
    if args.model_type.lower() == "lstm":
        run_lstm(loader, USE_DEV)


