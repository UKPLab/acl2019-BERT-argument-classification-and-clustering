# Argument Classification and Clustering using BERT
In our publication [Classification and Clustering of Arguments with Contextualized Word Embeddings]() (ACL 2019) we fine-tuned the BERT network to: 1) Perform sentential argument classification (i.e., given a sentence with an argument for a controversial topic, classify this sentence as pro, con, or no argument), 2) Estimate the argument similarity (0...1) given two sentences. This argument similarity score can be used in conjuction with hierarchical agglomerative clustering to perform aspect-based argument clustering.


# Citation
If you find the implementation useful, please cite the following paper: [Classification and Clustering of Arguments with Contextualized Word Embeddings]()

```
@InProceedings{Reimers:2019:ACL,
  author    = {Reimers, Nils, and Schiller, Benjamin and Beck, Tilman and Daxenberger, Johannes and Stab, Christian and Gurevych, Iryna},
  title     = {{Classification and Clustering of Arguments with Contextualized Word Embeddings}},
  booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  month     = {07},
  year      = {2019},
  address   = {Florence, Italy},
  pages     = {},
  url       = {}
}
``` 



Contact person: Nils Reimers, Rnils@web.de

https://www.ukp.tu-darmstadt.de/ https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

# Setup

This repository requires Python 3.5+ and PyTorch 0.4.1/1.0.0. It uses [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT/) version 0.6.2. See the [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT/) readme for further details on the installation. Usually, you can install as follows:
```
pip install pytorch-pretrained-bert==0.6.2 sklearn scipy
```





