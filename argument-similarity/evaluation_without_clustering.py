"""
Computes the F1-scores without clustering (Table 2 in the paper).
"""
import numpy as np
import csv
from sklearn.metrics import f1_score
from collections import defaultdict


class PairwisePredictionSimilarityScorer:
    def __init__(self, predictions_file):
        self.score_lookup = defaultdict(dict)
        for line in open(predictions_file):
            splits = line.strip().split('\t')
            score = float(splits[-1])
            sentence_a = splits[0].strip()
            sentence_b = splits[1].strip()
            self.score_lookup[sentence_a][sentence_b] = score
            self.score_lookup[sentence_b][sentence_a] = score


    def get_similarity(self, sentence_a, sentence_b):
        return self.score_lookup[sentence_a][sentence_b]




######################################
#
# Some help functions
#
######################################
def evaluate(similarity_score_function, labels_file, threshold, print_scores=False):
    all_f1_means = []
    all_f1_sim = []
    all_f1_dissim = []

    test_data = defaultdict(list)
    with open(labels_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
        for splits in csvreader:
            splits = map(str.strip, splits)
            label_topic, sentence_a, sentence_b, label = splits
            label_bin = '1' if label in ['SS', 'HS'] else '0'

            test_data[label_topic].append({'topic': label_topic, 'sentence_a': sentence_a, 'sentence_b': sentence_b, 'label': label,
                 'label_bin': label_bin})

    for topic in test_data:
        topic_test_data = test_data[topic]
        y_true = np.zeros(len(topic_test_data))
        y_pred = np.zeros(len(topic_test_data))

        for idx, test_annotation in enumerate(topic_test_data):
            sentence_a = test_annotation['sentence_a']
            sentence_b = test_annotation['sentence_b']
            label = test_annotation['label_bin']

            if label == '1':
                y_true[idx] = 1

            if similarity_score_function(sentence_a, sentence_b) > threshold:
                y_pred[idx] = 1




        f_sim = f1_score(y_true, y_pred, pos_label=1)
        f_dissim = f1_score(y_true, y_pred, pos_label=0)
        f_mean = np.mean([f_sim, f_dissim])
        all_f1_sim.append(f_sim)
        all_f1_dissim.append(f_dissim)
        all_f1_means.append(f_mean)

        if print_scores:
            print("F-Sim: %.2f%%" % (f_sim * 100))
            print("F-Dissim: %.2f%%" % (f_dissim * 100))
            print("F-Mean: %.2f%%" % (f_mean * 100))
            acc = np.sum(y_true==y_pred) / len(y_true)
            print("Acc: %.2f%%" % (acc * 100))

    return np.mean(all_f1_sim), np.mean(all_f1_dissim), np.mean(all_f1_means)



######################################
#
# Functions for pairwise classification approaches
#
######################################
def trained_pairwise_prediction_clustering(bert_experiment, epoch):

    print("Epoch:", epoch)

    all_f1_sim = []
    all_f1_dissim = []
    all_f1 = []
    for split in [0, 1, 2, 3]:
        print("\n==================")
        print("Split:", split)
        dev_file = './datasets/ukp_aspect/splits/%d/dev.tsv' % (split)
        test_file = './datasets/ukp_aspect/splits/%d/test.tsv' % (split)

        dev_sim_scorer = PairwisePredictionSimilarityScorer("%s/%d/dev_predictions_epoch_%d.tsv" % (bert_experiment, split, epoch))
        test_sim_scorer = PairwisePredictionSimilarityScorer("%s/%d/test_predictions_epoch_%d.tsv" % (bert_experiment, split, epoch))

        best_f1 = 0
        best_threshold = 0

        for threshold_int in range(0, 20):
            threshold = threshold_int / 20
            f1_sim, f1_dissim, f1 = evaluate(dev_sim_scorer.get_similarity, dev_file, threshold)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print("Best threshold on dev:", best_threshold, "F1:", best_f1)

        # Evaluate on test
        f1_sim, f1_dissim, f1 = evaluate(test_sim_scorer.get_similarity, test_file, best_threshold)

        all_f1_sim.append(f1_sim)
        all_f1_dissim.append(f1_dissim)
        all_f1.append(f1)

        print("Test-Performance on this split:")
        print("F-Mean: %.4f" % (f1))
        print("F-sim: %.4f" % (f1_sim))
        print("F-dissim: %.4f" % (f1_dissim))



    print("\n\n===========  Averaged performance over all splits ==========")
    print("F-Mean: %.4f" % (np.mean(all_f1)))
    print("F-sim: %.4f" % (np.mean(all_f1_sim)))
    print("F-dissim: %.4f" % (np.mean(all_f1_dissim)))
    return np.mean(all_f1)


def main():
    bert_experiment = 'bert_output/ukp/seed-1/splits/'
    trained_pairwise_prediction_clustering(bert_experiment, epoch=2)


if __name__ == '__main__':
    main()