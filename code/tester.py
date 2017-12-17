from sklearn.metrics import f1_score
import numpy as np

def transform_one_hot(data, mapping, length):
    new_seqs = []
    for sequence in data:
        new_seq = []
        for tag in sequence:
            max_index = np.argmax(tag)
            tag_one_hot = np.zeros((length))
            tag_one_hot[max_index] += 1
            new_seq.append(mapping[tuple(tag_one_hot)])
        new_seqs.append(new_seq)

    return new_seqs

def evaluate_f1(model, loader, dev):
    if dev:
        y_pred = model.predict(loader.X_dev)
    else:
        y_pred = model.predict(loader.X_test)

    print y_pred.shape

    y_pred_tags = transform_one_hot(y_pred, loader.one_hot_to_tag, loader.tag_length)
    if dev:
        y_true_tags = transform_one_hot(loader.Y_dev, loader.one_hot_to_tag, loader.tag_length)
    else:
        y_true_tags = transform_one_hot(loader.Y_test, loader.one_hot_to_tag, loader.tag_length)

    # false positive: times we classified something as <tag> when it wasn't <tag>
    # false negative: times we didn't classify something as <tag> but it was <tag> 
    # true postivie: times we classified something as <tag> and it was <tag>  
    # true negative: times we classified something as anything but <tag> and it wasn't <tag> 

    for true, predicted in zip(y_pred_tags, y_true_tags):
        





    return f1_score(y_true_tags, y_pred_tags)
