from sklearn.metrics import f1_score
import numpy as np
import sys
import re 


def transform_one_hot(data, mapping, length, sampling):
    """
    transform a predicted tag vector to a tag label by turning the predicted vector into
    a one-hot, either via sampling or via argmax 

    Parameters
    ----------

    Returns
    -------


    """
    new_seqs = []
    for sequence in data:
        new_seq = []
        for tag in sequence:
            if sampling:
                tag_one_hot = np.random.multinomial(1, tag)
            else:
                max_index = np.argmax(tag)
                tag_one_hot = np.zeros((length))
                tag_one_hot[max_index] += 1
            new_seq.append(mapping[tuple(tag_one_hot)])
        new_seqs.append(new_seq)

    return new_seqs


def get_spans(sequences):
    """return a sequence of spans where a span is defined as everything before the O in a
     sequence like B-Tag ... I-Tag O

    Parameters
    ----------
    sequence: list of sequences
        ground truth or predicted sequences

    Returns
    -------
    spans: list of tuples 
        each tuple of type (tag_type, begin_index, end_index)
    """

    begin_regex = re.compile(r"(?<=(B-))(.*)")
    inside_regex_string = r"(?<=(I-))({})"

    all_spans = []
    for seq in sequences: 
        i=0
        spans = []
        while i<len(seq):
            tag = seq[i]
            m = begin_regex.search(tag)
            if m is not None:
                tag_type = m.group(0)
                inside_regex = re.compile(inside_regex_string.format(tag_type))
                j = i+1
                try:
                    m2 = inside_regex.search(seq[j])
                except IndexError:
                    break
                current_span = list((tag_type, i, i))
                while j<len(seq) and m2 is not None:
                    current_span[2] = j
                    j+=1
                    try:
                        m2 = inside_regex.search(seq[j])
                    except IndexError:
                        break
                spans.append(current_span)
                i = j-1
            i+=1
        all_spans.append(spans)
    return all_spans

def span_equals(span1, span2):
    for e1, e2 in zip(span1, span2):
        if e1 != e2:
            return False
    return True


def conll_f1(true_spans, pred_spans):
    # from Conll2003 paper:
    # Precision is the percentage of named entities found by the learning system that are correct. 
    # Recall is the percentage of named entities present in the corpus that are found by the system
    precision = 0
    recall = 0
    total_predicted = 0
    total_true = 0

    for true, pred in zip(true_spans, pred_spans):
        total_predicted += len(pred)
        total_true += len(true)
        for i, t_span in enumerate(true):
            for j, p_span in enumerate(pred):
                if span_equals(t_span, p_span):
                    recall += 1
        for i, p_span in enumerate(pred):
            for j, t_span in enumerate(true):
                if span_equals(p_span, t_span):
                    precision += 1

    recall = float(recall)/float(total_true)
    precision = float(precision)/float(total_predicted)

    f1 = 2*(precision*recall)/(precision+recall)

    return f1


def evaluate_f1(model, loader, dev, sampling=False):
    if dev:
        y_pred = model.predict(loader.X_dev)
    else:
        y_pred = model.predict(loader.X_test)

    y_pred_tags = transform_one_hot(y_pred, loader.one_hot_to_tag, loader.tag_length, sampling)
    if dev:
        y_true_tags = transform_one_hot(loader.Y_dev, loader.one_hot_to_tag, loader.tag_length, sampling)
    else:
        y_true_tags = transform_one_hot(loader.Y_test, loader.one_hot_to_tag, loader.tag_length, sampling)

    true_spans = get_spans(y_true_tags)
    pred_spans = get_spans(y_pred_tags)
    f1 = conll_f1(true_spans, pred_spans)
    # sanity_check = conll_f1(true_spans, true_spans)
    # print "sanity check f1: {}".format(sanity_check)

    return f1
