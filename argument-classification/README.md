# Argument Classification
This folder contains code to fine-tune BERT for argument classification: Is a given sentence a pro, con, or no argument for a given topic?

**Example:**
```
Topic: zoo
Sentence: Zoo confinement is psychologically damaging to animals.
Output Label: Argument_against
```


## Setup
For the setup, see the [README.md](https://github.com/UKPLab/acl2019-BERT-argument-classification-and-clustering/) in the main folder.


## Training
We trained (and evaluated) our models on the [UKP Sentential Argument Mining Corpus](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/ukp_sentential_argument_mining_corpus/index.en.jsp), which annotated 25,492 sentences over eight controversial topics.

Due to copyright issues, we cannot distribute the corpus directly. You need to download it from the website and run a Java program, to re-construct the corpus from the sources. More information can be found in the README.txt of the UKP Sentential Argument Mining Corpus.

Once you have re-created the UKP Sentential Argument Mining Corpus, you can fine-tune BERT by running the `train_ukp.sh` script:
```
./train_ukp.sh
```

This fine-tunes BERT on seven topics and evaluates the performance on the eigth topic.


If you want to train BERT on all 25,492 sentences from the UKP Argument Corpus, run `train_ukp_all_data.sh`.

We also provide a data reader and script for the [IBM Debater dataset](http://www.research.ibm.com/haifa/dept/vst/debating_data.shtml). See `train_ibm.sh` how to train BERT for this dataset. As before, you first need to download the corpus and unzip it to `datasets/ibm/.

**Note:** Training on GPU leads to non-determinisitc results. For scientific experiments, we recommend to train with multiple random seeds and to average results.

## Inference
You can use `inference.py` to classify new arguments on new topics:
```
python inference.py
```

You must specify the model path:
```
model_path = 'bert_output/ukp/bert-base-topic-sentence/all_topics/'
```

Download and unzip a pre-trained model from here:
- [argument_classification_ukp_all_data.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip)
- [argument_classification_ukp_all_data_large_model.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data_large_model.zip)

This model was trained on all eight topics of the [UKP Sentential Argument Mining Corpus](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/ukp_sentential_argument_mining_corpus/index.en.jsp). The topics are: bortion, cloning, death penalty, gun control, marijuana legalization, minimum wage, nuclear energy, school uniforms.

This model can be applied for arguments from different topics, for example, for keeping animals in zoos (this topic was not in the training data):
```
Predicted labels:
Topic: zoo
Sentence: A zoo is a facility in which all animals are housed within enclosures, displayed to the public, and in which they may also breed.
Gold label: NoArgument
Predicted label: NoArgument

Topic: zoo
Sentence: Zoos produce helpful scientific research.
Gold label: Argument_for
Predicted label: Argument_for

Topic: zoo
Sentence: Zoos save species from extinction and other dangers.
Gold label: Argument_for
Predicted label: Argument_for

Topic: zoo
Sentence: Zoo confinement is psychologically damaging to animals.
Gold label: Argument_against
Predicted label: Argument_against

Topic: zoo
Sentence: Zoos are detrimental to animals' physical health.
Gold label: Argument_against
Predicted label: Argument_against

Topic: autonomous cars
Sentence: Zoos are detrimental to animals' physical health.
Gold label: NoArgument
Predicted label: NoArgument
```

Note, when you change the topic for an argument, as in the last example, the model corretly identifies that this sentence is not an argument for / against 'autonomous cars'.



## Performance

In a cross-topic evaluation, the BERT model achieves the following performance.

![Classification Performance](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/images/table_classification_results.png)


See our paper ([Classification and Clustering of Arguments with Contextualized Word Embeddings](https://arxiv.org/abs/1906.09821))  for further details.

For the computation of the macro F1-score for the UKP corpus, see `ukp_evaluation.py`.
