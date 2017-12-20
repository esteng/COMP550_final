from data import DataLoader
from tester import evaluate_f1, evaluate_accuracy

import sys


loader = DataLoader()
path = sys.argv[1]
sentences, tup = loader.get_most_common(path, 0)
top, other = tup
top_tag = top[0]
other_tag = other[0]

loader.get_file_data(path, None)



        

print "top tag: ",top_tag
print "other tag: ", other_tag

y_pred_top = len(sentences)*[30*[loader.tag_to_one_hot[top_tag]]]
y_pred_other = len(sentences)*[30*[loader.tag_to_one_hot[other_tag]]]

class Placeholder(object):
    """docstring for Placeholder"""
    def __init__(self, y_pred_other, y_pred_top):
        super(Placeholder, self).__init__()
        self.y_pred_other = y_pred_other
        self.y_pred_top = y_pred_top
    def predict(_, waste):
        
        return y_pred_top

model = Placeholder(y_pred_other, y_pred_top)

f1 = evaluate_f1(model, loader, False)


acc = evaluate_accuracy(model, loader, False)





