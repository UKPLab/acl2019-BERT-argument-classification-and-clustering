# Argument Classification and Clustering using BERT
In our publication [Classification and Clustering of Arguments with Contextualized Word Embeddings](https://arxiv.org/abs/1906.09821) (ACL 2019) we fine-tuned the BERT network to:
- Perform sentential argument classification (i.e., given a sentence with an argument for a controversial topic, classify this sentence as pro, con, or no argument). Details can be found in [argument-classification/README.md](argument-classification/README.md)
- Estimate the argument similarity (0...1) given two sentences. This argument similarity score can be used in conjuction with hierarchical agglomerative clustering to perform aspect-based argument clustering. Details can be found in [argument-similarity/README.md](argument-similarity/README.md)


# Citation
If you find the implementation useful, please cite the following paper: [Classification and Clustering of Arguments with Contextualized Word Embeddings](https://arxiv.org/abs/1906.09821)

```
@InProceedings{Reimers:2019:ACL,
  author    = {Reimers, Nils, and Schiller, Benjamin and Beck, Tilman and Daxenberger, Johannes and Stab, Christian and Gurevych, Iryna},
  title     = {{Classification and Clustering of Arguments with Contextualized Word Embeddings}},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {07},
  year      = {2019},
  address   = {Florence, Italy},
  pages     = {567--578},
  url       = {https://arxiv.org/abs/1906.09821}
}
``` 



Contact person: Nils Reimers, Rnils@web.de

https://www.ukp.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

# Setup

This repository requires Python 3.5+ and PyTorch 0.4.1/1.0.0. It uses [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT/) version 0.6.2. See the [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT/) readme for further details on the installation. Usually, you can install as follows:
```
pip install pytorch-pretrained-bert==0.6.2 sklearn scipy
```

# Argument Classification
Please see [argument-classification/README.md](argument-classification/README.md) for full details.

Given a sentence and a topic, classify if the sentence is a pro, con, or no argument. For example:
```
Topic: zoo
Sentence: Zoo confinement is psychologically damaging to animals.
Output Label: Argument_against
```

You can download pre-trained models from here, which were trained on all eight topics of the [UKP Sentential Argument Mining Corpus](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/ukp_sentential_argument_mining_corpus/index.en.jsp):
- [argument_classification_ukp_all_data.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data.zip)
- [argument_classification_ukp_all_data_large_model.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_classification_ukp_all_data_large_model.zip)


See [argument-classification/inference.py](argument-classification/inference.py) how to use these models for classifying new sentences.

In a leave-one-topic out evaluation, the BERT model achieves the following performance.

![Classification Performance](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/images/table_classification_results.png)


# Argument Similarity & Clustering
See [argument-similarity/README.md](argument-similarity/README.md) for full details.

Given two sentences, the code in [argument-similarity](argument-similarity/) returns a value between 0 and 1 indicating the similarity between the arguments. This can be used for agglomorative clustering to find & cluster similar arguments.

You can download two pre-trained models:
- [argument_similarity_ukp_aspects_all.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_similarity_ukp_aspects_all.zip) - trained on the complete [UKP Argument Aspect Similarity Corpus](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/argumentation_mining_1/ukp_argument_aspect_similarity_corpus/ukp_argument_aspect_similarity_corpus.en.jsp)
- [argument_similarity_misra_all.zip](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/models/argument_similarity_misra_all.zip) - trained on the complete [Argument Facet Similarity (AFS) Corpus](https://nlds.soe.ucsc.edu/node/44) from Misra et al.


See [argument-similarity/inference.py](argument-similarity/inference.py) for an example. This example computes the pairwise similarity between arguments on different topics.
The output should be something like this for the model trained on the UKP corpus:
```
Predicted similarities (sorted by similarity):
Sentence A: Eating meat is not cruel or unethical; it is a natural part of the cycle of life.
Sentence B: It is cruel and unethical to kill animals for food when vegetarian options are available
Similarity: 0.99436545

Sentence A: Zoos are detrimental to animals' physical health.
Sentence B: Zoo confinement is psychologically damaging to animals.
Similarity: 0.99386144

[...]

Sentence A: It is cruel and unethical to kill animals for food when vegetarian options are available
Sentence B: Rising levels of human-produced gases released into the atmosphere create a greenhouse effect that traps heat and causes global warming.
Similarity: 0.0057242378
```

With the Misra AFS model, the output should be something like this:
```
Predicted similarities (sorted by similarity):
Sentence A: Zoos are detrimental to animals' physical health.
Sentence B: Zoo confinement is psychologically damaging to animals.
Similarity: 0.8723387

Sentence A: Eating meat is not cruel or unethical; it is a natural part of the cycle of life.
Sentence B: It is cruel and unethical to kill animals for food when vegetarian options are available
Similarity: 0.77635074

[...]

Sentence A: Zoos produce helpful scientific research.
Sentence B: Eating meat is not cruel or unethical; it is a natural part of the cycle of life.
Similarity: 0.20616204
```


## Argument Similarty Performance

![UKP Aspects Performance](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/images/table_UKP_Aspects_results.png)

![AFS Performance](https://public.ukp.informatik.tu-darmstadt.de/reimers/2019_acl-BERT-argument-classification-and-clustering/images/table_AFS_results.png)




