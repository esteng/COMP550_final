import subprocess
import itertools
# parser.add_argument("--input_file", dest="input_path", required=True,
#     help="mandatory path to NER corpus, either in csv or txt format")
# parser.add_argument("--model_type", dest="model_type", required=True, type=str, 
#     help="mandatory argument: what type of NER system to run. \n Expected: 'lstm' or 'hmm'")
# parser.add_argument("--stop", dest="stop", required=False, default=None, 
#     help="whether or not to remove stopwords")
# # LSTM arguments
# parser.add_argument("--embedding_file", dest='embedding_path', 
#     default=None, required = False,
#     help="optional path to an embedding file. If left blank, no pre-trained embeddings will be used")
# parser.add_argument("--resume_model", dest="resume_path", required=False, type=str, 
#     help="resume LSTM model from existing file")
# parser.add_argument("--layer_num", dest="layer_num", required=False, default=1, 
#     help="the number of bidirectional LSTM and dropout layer pairs in the model")
# # HMM arguments
# parser.add_argument("--estimator", dest="estimator", default="Laplace", required=False, type=str, 
#     help="if using an hmm, choose an estimator \n Expected: 'mle' or 'Laplace'")
# parser.add_argument("--experiment", dest="experiment", default="1", required=True, type=str,
#     help="Choose experiment 1 or 2 (full train-test or small train-test)")

input_files = ["ned", "esp", "../data/entity-annotated-corpus/ner_dataset.csv"]
model_types = ["lstm, hmm"]
lstm = ["lstm"]
embedding_files = ["../embeddings/SBW-vectors-300-min5.bin"]
estimators = ["mle", "Laplace"]
experiment = ["1", "2"]
hmm = ["hmm"]

for lstm_combination in itertools.product(lstm, input_files, experiment):
    if lstm_combination[1] == "esp":
        embedding_file = embedding_files[0]
        try:

            subprocess.Popen(["python", "run.py", "--input_file=",lstm_combination[1],
         "--model_type=", lstm_combination[0], "--embedding_file=", embedding_file, 
         "--experiment=", lstm_combination[2]], shell=True).communicate()
        except:
            print "mistake: "
            print lstm_combination
    else:
        try:
            subprocess.Popen(["python", "run.py", "--input_file=",lstm_combination[1],
         "--model_type=", lstm_combination[0], "--experiment=", lstm_combination[2]], shell=True).communicate()
        except:
            print "mistake:"
            print lstm_combination

for hmm_combo in itertools.product(hmm, input_files, experiment, estimators):
    try:
        subprocess.Popen(["python", "run.py", "--model_type=", "hmm", "--input_file=", hmm_combo[1], 
            "--experiment=", hmm_combo[2], "--estimator=", hmm_combo[3]]).communicate()
    except: 
        print "mistake: "
        print hmm_combo






