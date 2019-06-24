"""
Evaluates the performance on the UKP ASPECT Corpus with hierachical clustering (Table 2 in our paper).

Greedy hierachical clustering.
Merges two clusters if the pairwise mean cluster similarity is larger than a threshold.
Merges clusters with highest similarity first
Uses dev set to determine the threshold for supervised systems
"""
import numpy as np
import scipy
import scipy.spatial.distance
import csv
import os
from sklearn.metrics import f1_score
from collections import defaultdict

class VectorSimilarityScorer:
    def __init__(self, sentence_vectors):
        self.vector_lookup = {}

        for line in open(sentence_vectors):
            sentence, vector_str = line.strip().split('\t')
            vector = np.asarray(list(map(float, vector_str.split(" "))))
            self.vector_lookup[sentence] = vector

        self.cache = defaultdict(dict)

    def get_similarity(self, sentence_a, sentence_b):
        if sentence_a not in self.cache or sentence_b not in self.cache[sentence_a]:
            vector_a = self.vector_lookup[sentence_a]
            vector_b = self.vector_lookup[sentence_b]
            cosine_sim = 1 - scipy.spatial.distance.cosine(vector_a, vector_b)
            self.cache[sentence_a][sentence_b] = cosine_sim
            self.cache[sentence_b][sentence_a] = cosine_sim
        return self.cache[sentence_b][sentence_a]


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

class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.queue.append(data)

    # Removes all element addressing a cluster key
    def remove_clusters(self, cluster_key):
        i = 0
        while i < len(self.queue):
            ele = self.queue[i]
            if ele['cluster_a'] == cluster_key or ele['cluster_b'] == cluster_key:
                del self.queue[i]
            else:
                i += 1

    # for popping an element based on Priority
    def pop(self):
        max = 0
        for i in range(len(self.queue)):
            if self.queue[i]['cluster_sim'] > self.queue[max]['cluster_sim']:
                max = i
        item = self.queue[max]
        del self.queue[max]
        return item


class HierachicalClustering:
    """
    Simple clustering algorithm. Merges two clusters, if the cluster similarity is larger than the threshold.
    Highest similarities first.
    """
    def __init__(self, similarity_score_function, testfile, np_mode=np.mean):
        self.compute_similarity_score = similarity_score_function
        self.test_data, self.clusters = self.read_gold_data(testfile)
        self.np_mode = np_mode

    def read_gold_data(self, testfile):
        test_data = {}
        unique_sentences = {}

        with open(testfile, 'r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter='\t', quotechar=None)
            for splits in csvreader:
                splits = map(str.strip, splits)
                topic, sentence_a, sentence_b, label = splits
                label_bin = '1' if label in ['SS', 'HS'] else '0'

                if topic not in test_data:
                    test_data[topic] = []

                test_data[topic].append({'topic': topic, 'sentence_a': sentence_a, 'sentence_b': sentence_b, 'label': label,
                                         'label_bin': label_bin})

                if topic not in unique_sentences:
                    unique_sentences[topic] = set()

                unique_sentences[topic].add(sentence_a)
                unique_sentences[topic].add(sentence_b)

        cluster_info = {}
        for topic in unique_sentences:
            topic_sentences = unique_sentences[topic]
            cluster_info[topic] = {}
            for idx, sentence in enumerate(topic_sentences):
                cluster_info[topic][idx] = [sentence]

        return test_data, cluster_info



    def compute_cluster_sim(self, cluster_a, cluster_b):
        scores = []
        for sentence_a in cluster_a:
            for sentence_b in cluster_b:
                scores.append(self.compute_similarity_score(sentence_a, sentence_b))

        return self.np_mode(scores)

    def cluster_topics(self, threshold):
        for topic in self.clusters:
            #print("\nRun clustering for:", topic)
            topic_cluster = self.clusters[topic]
            self.run_clustering(topic_cluster, threshold)
        return self.clusters



    def run_clustering(self, clusters, threshold):
        queue = PriorityQueue()

        #Initial cluster sim computation
        cluster_ids = list(clusters.keys())
        for i in range(0, len(cluster_ids)-1):
            for j in range(i+1, len(cluster_ids)):
                cluster_a = cluster_ids[i]
                cluster_b = cluster_ids[j]

                cluster_sim = self.compute_cluster_sim(clusters[cluster_a], clusters[cluster_b])
                element = {'cluster_sim': cluster_sim, 'cluster_a': cluster_a, 'cluster_b': cluster_b}
                queue.insert(element)

        while not queue.isEmpty():
            element = queue.pop()
            if element['cluster_sim'] <= threshold:
                break

            #print("Merge", element, "size_a:", len(clusters[element['cluster_a']]), "size_b:", len(clusters[element['cluster_b']]))
            #Merge cluster with highest sim
            self.merge_clusters(clusters, element['cluster_a'], element['cluster_b'])

            #Remove all element involving cluster_a or cluster_b
            queue.remove_clusters(element['cluster_a'])
            queue.remove_clusters(element['cluster_b'])

            #Recompute cluster sim for all clusters with cluster_a and cluster_b
            cluster_a = element['cluster_a']
            for cluster_b in clusters.keys():
                if cluster_a != cluster_b:
                    cluster_sim = self.compute_cluster_sim(clusters[cluster_a], clusters[cluster_b])
                    element = {'cluster_sim': cluster_sim, 'cluster_a': cluster_a, 'cluster_b': cluster_b}
                    queue.insert(element)

    def merge_clusters(self, clusters, key_a, key_b):
        clusters[key_a] += clusters[key_b]
        del clusters[key_b]


######################################
#
# Some help functions
#
######################################

def get_clustering(similarity_function, testfile, threshold):
    cluster_alg = HierachicalClustering(similarity_function, testfile)
    clusters = cluster_alg.cluster_topics(threshold)
    return clusters

def write_output_file(clusters, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    fOut = open(output_file, 'w')

    for topic in clusters:
        topic_cluster = clusters[topic]
        for cluster_id in topic_cluster:
            for sentence in topic_cluster[cluster_id]:
                fOut.write("\t".join([str(cluster_id), topic, sentence.replace("\n", " ").replace("\t", " ")]))
                fOut.write("\n")


def evaluate(clusters, labels_file, print_scores=False):
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


            test_data[label_topic].append(
                    {'topic': label_topic, 'sentence_a': sentence_a, 'sentence_b': sentence_b, 'label': label,
                     'label_bin': label_bin})

    for topic in clusters:
        topic_cluster = clusters[topic]
        sentences_cluster_id = {}
        for cluster_id in topic_cluster:
            for sentence in topic_cluster[cluster_id]:
                sentences_cluster_id[sentence] = cluster_id

        topic_test_data = test_data[topic]


        y_true = np.zeros(len(topic_test_data))
        y_pred = np.zeros(len(topic_test_data))

        for idx, test_annotation in enumerate(topic_test_data):
            sentence_a = test_annotation['sentence_a']
            sentence_b = test_annotation['sentence_b']
            label = test_annotation['label_bin']

            if label=='1':
                y_true[idx] = 1

            if sentences_cluster_id[sentence_a] == sentences_cluster_id[sentence_b]:
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
        output_file = None #'output/bert-base-uncased/%s/seed-%d/%d/test_clusters.tsv' % (transitive, seed, split)

        dev_sim_scorer = PairwisePredictionSimilarityScorer("%s/%d/dev_predictions_epoch_%d.tsv" % (bert_experiment, split, epoch))
        test_sim_scorer = PairwisePredictionSimilarityScorer("%s/%d/test_predictions_epoch_%d.tsv" % (bert_experiment, split, epoch))

        best_f1 = 0
        best_threshold = 0

        for threshold_int in range(0, 20):
            threshold = threshold_int / 20
            clusters = get_clustering(dev_sim_scorer.get_similarity, dev_file, threshold)
            f1_sim, f1_dissim, f1 = evaluate(clusters, dev_file)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print("Best threshold on dev:", best_threshold)

        # Evaluate on test
        clusters = get_clustering(test_sim_scorer.get_similarity, test_file, best_threshold)
        if output_file != None:
            write_output_file(clusters, output_file)
        f1_sim, f1_dissim, f1 = evaluate(clusters, test_file)

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
    bert_experiment = 'bert_output/ukp/seed-1/splits'
    trained_pairwise_prediction_clustering(bert_experiment, epoch=3)


if __name__ == '__main__':
    main()