import re
import argparse
import sys
import csv

from base_lstm import define_model
from data import DataLoader
from tester import evaluate_f1

# embedding_size, tag_size, input_length, embedding = False, vocab_dim = 0
def run_lstm(loader, output_file, use_dev, layer_num):
    print "defining model..."
    if loader.use_embedding_layer:
        model = define_model(loader.w2v_size, loader.tag_length, loader.max_length, 
            layer_num, True, loader.vocab_size)
    else:
        model = define_model(loader.w2v_size, loader.tag_length, loader.max_length, layer_num)

    print "model summary:"
    print model.summary()

    print "training..." 
    model.fit(loader.X_train, loader.Y_train, epochs=2, batch_size=128, verbose=1,  shuffle=True)

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

if __name__ == '__main__':
    USE_DEV = True
    LAYER_NUM = 2
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
    args = parser.parse_args()

    loader = DataLoader()

    
    loader.get_file_data(args.input_path, args.embedding_path)
    
    if args.model_type.lower() == "lstm":
        run_lstm(loader, "../results/lstm_results.csv", USE_DEV, LAYER_NUM)


