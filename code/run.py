import re
from base_lstm import define_model
from data import DataLoader
import argparse


def run_lstm(X_train, Y_train, X_test, Y_test, w2v_size, tag_length, max_length):
    print("defining model...")
    model = define_model(w2v_size, tag_length, max_length)
    print "model summary:"
    print model.summary()
    print("training...")
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1,  shuffle=True)
    print "evaluating"
    results = model.evaluate(X_test, Y_test, verbose=1)
    return results

if __name__ == '__main__':
    print("in main")
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
        run_lstm(loader.X_train, loader.Y_train, loader.X_dev, loader.Y_dev, 
            loader.w2v_size, loader.tag_length, loader.max_length)


