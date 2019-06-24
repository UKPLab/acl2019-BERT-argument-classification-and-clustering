# This code runs the macro-weighted F1-computation for the UKP  Sentential Argument Mining Corpus (Table 1 in the Paper Classification and Clustering of Arguments with Contextualized Word Embeddings)
# Usage: python ukp_evaluation.py bert_output/ukp/bert-base-topic-sentence/*/test_predictions.txt
# Expected output:
#   ...
#   ===================== Overall results ===================="
#   Average F1 score over all topics
#   test_predictions.txt (8) 61.69%
#
# Note: output values can change due to the non-determinism of GPU computations


from __future__ import print_function
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.utils.multiclass import unique_labels
import numpy as np
import sys
import os

def analyze_predictions(filepath):
    total_sent = 0
    correct_sent = 0
    count = {}

    y_true = []
    y_pred = []

    for line in open(filepath, encoding='utf8'):
        splits = line.strip().split("\t")
        gold = splits[0]
        pred = splits[1]

        total_sent += 1
        if gold == pred:
            correct_sent += 1

        if gold not in count:
            count[gold] = {}

        if pred not in count[gold]:
            count[gold][pred] = 0

        count[gold][pred] += 1

        y_true.append(gold)
        y_pred.append(pred)

    print("gold - pred - Confusion Matrix")
    for gold_label in sorted(count.keys()):
        for pred_label in sorted(count[gold_label].keys()):
            print("%s - %s: %d" % (gold_label, pred_label, count[gold_label][pred_label]))


    print(":: BERT ::")
    print("Acc: %.2f%%" % (correct_sent/total_sent*100) )
    labels = unique_labels(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)

    arg_f1 = []
    for idx, label in enumerate(labels):
        print("\n:: F1 for "+label+" ::")
        print("Prec: %.2f%%" % (prec[idx]*100))
        print("Recall: %.2f%%" % (rec[idx]*100))
        print("F1: %.2f%%" % (f1[idx]*100))

        if label in labels:
            if label != 'NoArgument':
                arg_f1.append(f1[idx])


    print("\n:: Macro Weighted for all  ::")
    print("F1: %.2f%%" % (np.mean(f1)*100))

    prec_mapping = {key:value for key, value in zip(labels, prec)}
    rec_mapping = {key:value for key, value in zip(labels, rec)}
    return np.mean(f1), prec_mapping, rec_mapping

results = {}
prec_results = {}
rec_results = {}
for filepath in sys.argv[1:]:
    print("\n\n===================== "+filepath+" ====================")
    f1, prec, rec = analyze_predictions(filepath)

    folder = filepath.split('/')[-2]
    topic = folder.split('_')[0].split('(')[0]
    filename = os.path.basename(filepath)

    if topic not in results:
        results[topic] = {}
        prec_results[topic] = {}
        rec_results[topic] = {}

    if filename not in results[topic]:
        results[topic][filename] = []
        prec_results[topic][filename] = []
        rec_results[topic][filename] = []


    results[topic][filename].append(f1)
    prec_results[topic][filename].append(prec)
    rec_results[topic][filename].append(rec)



print("\n\n===================== Overall results ====================")
model_f1 = {}
model_prec = {}
model_rec = {}

for topic in sorted(results.keys()):
    print(topic)
    for filename in sorted(results[topic].keys()):
        topic_f1_mean = np.mean(results[topic][filename])
        print("%s (%d): %.4f" % (filename, len(results[topic][filename]), topic_f1_mean))

        if filename not in model_f1:
            model_f1[filename] = []
            model_prec[filename] = []
            model_rec[filename] = []

        model_f1[filename].append(results[topic][filename])
        for prec in prec_results[topic][filename]:
            model_prec[filename].append(prec)

        for rec in rec_results[topic][filename]:
            model_rec[filename].append(rec)
    print("")


print("\n\n==========================================")
print("Average F1 score over all topics")
for filename in model_f1:
    print("%s (%d) %.2f%%" % (filename, len(model_f1[filename]), np.mean(model_f1[filename])*100))


print("\n\n==========================================")
print("P_arg score over all topics")
for filename in model_prec:
    prec_pos = [prec_result['Argument_for'] for prec_result in model_prec[filename]]
    print("P_arg+ %s (%d): %.4f" % (filename, len(prec_pos), np.mean(prec_pos)))

    prec_neg = [prec_result['Argument_against'] for prec_result in model_prec[filename]]
    print("P_arg- %s (%d): %.4f" % (filename, len(prec_neg), np.mean(prec_neg)))

print("\n\n==========================================")
print("R_arg score over all topics")
for filename in model_rec:
    rec_pos = [rec_result['Argument_for'] for rec_result in model_rec[filename]]
    print("R_arg+ %s (%d): %.4f" % (filename, len(rec_pos), np.mean(rec_pos)))

    rec_neg = [rec_result['Argument_against'] for rec_result in model_rec[filename]]
    print("R_arg- %s (%d): %.4f" % (filename, len(rec_neg), np.mean(rec_neg)))