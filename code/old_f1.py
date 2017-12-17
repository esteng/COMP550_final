
    # false positive: times we classified something as <tag> when it wasn't <tag>
    # false negative: times we didn't classify something as <tag> but it was <tag> 
    # true postivie: times we classified something as <tag> and it was <tag>  
    # true negative: times we classified something as anything but <tag> and it wasn't <tag> 
    all_tags = loader.tag_to_one_hot.keys()
    false_pos = {x:0 for x in all_tags}
    false_neg = {x:0 for x in all_tags}
    true_pos =  {x:0 for x in all_tags}


    for k, tup in enumerate(zip(y_pred_tags, y_true_tags)):
        true, predicted = tup
        print loader.all_sents[k]
        print true

        for i, true_tag in enumerate(true):
            pred_tag = predicted[i]
            if true_tag == pred_tag:
                true_pos[true_tag] += 1
            else: 
                false_pos[pred_tag] += 1
                false_neg[true_tag] += 1

    # remove 0's 
    for dict_type in [false_pos, false_neg, true_pos]:
        for key,value in dict_type.items():
            if value == 0:
                dict_type[key] = 1


    precision = {tag:float(true_pos[tag])/float(true_pos[tag]+false_pos[tag]) for tag in all_tags}
    recall = {tag:float(true_pos[tag])/float(true_pos[tag]+false_neg[tag]) for tag in all_tags}

    f1_dict = {tag:(2*precision[tag]*recall[tag])/(precision[tag] + recall[tag]) for tag in all_tags}
    total_guesses = {tag:float(false_pos[tag] + false_neg[tag] + true_pos[tag]) for tag in all_tags}
    all_total = sum(total_guesses.values())
    guess_percent = {tag:total_guesses[tag]/float(all_total) for tag in all_tags}

    f1_score = sum([guess_percent[tag]*f1_dict[tag] for tag in all_tags])

    return f1_score

    # return f1_score(y_true_tags, y_pred_tags)
