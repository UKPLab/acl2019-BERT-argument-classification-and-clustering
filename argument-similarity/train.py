# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import gzip
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import scipy.stats
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from collections import defaultdict

from SigmoidBERT import SigmoidBERT

logging.basicConfig(format='%(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text_a, text_b=None, label=None, guid=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id




class UKPAspectsProcessor(object):
    def _read_dataset(self, filepath):
        sentences = defaultdict(lambda: defaultdict(dict))

        with open(filepath, 'r') as fIn:
            for line in fIn:
                splits = line.strip().split('\t')
                assert len(splits)==4

                topic = splits[0].strip()
                sentence_a = splits[1].strip()
                sentence_b = splits[2].strip()
                label = splits[-1].strip()

                #Binarize the label
                bin_label = 1 if label in ['SS', 'HS'] else 0

                sentences[topic][sentence_a][sentence_b] = bin_label
                sentences[topic][sentence_b][sentence_a] = bin_label

        return sentences

    def get_examples(self, data_dir, train_file, dev_file, test_file, data_set):
        """See base class."""
        logging.info("Get "+ data_set+ " examples")

        if data_set == 'train':
            data_file = os.path.join(data_dir, train_file)
            return self._get_train_examples(data_file, data_set)
        elif data_set == 'dev':
            data_file = os.path.join(data_dir, dev_file)
            return self._get_test_examples(data_file, data_set)
        else:
            data_file = os.path.join(data_dir, test_file)
            return self._get_test_examples(data_file, data_set)

    def _get_train_examples(self, data_file, data_set):
        sentences = self._read_dataset(data_file)
        topics = list(sentences.keys())
        random.shuffle(topics)

        examples = []
        logging.info("Topics: " + str(topics))

        for topic in topics:
            for sentence_a in sentences[topic].keys():
                for sentence_b in sentences[topic][sentence_a].keys():
                    guid = "%s-%d" % (data_set, len(examples))
                    label = sentences[topic][sentence_a][sentence_b]
                    examples.append(InputExample(guid=guid, text_a=sentence_a, text_b=sentence_b, label=label))

        return examples


    def _get_test_examples(self, data_file, data_set):
        sentences = self._read_dataset(data_file)
        topics = list(sentences.keys())
        logging.info("Topics: "+str(topics))

        examples = []
        for topic in topics:
            unique_sentences = list(sentences[topic].keys())
            for i in range(len(unique_sentences)-1):
                for j in range(i+1, len(unique_sentences)):
                    guid = "%s-%d" % (data_set, len(examples))
                    sentence_a = unique_sentences[i]
                    sentence_b = unique_sentences[j]
                    label = -1

                    if sentence_b in sentences[topic][sentence_a]:
                        label = sentences[topic][sentence_a][sentence_b]

                    examples.append(InputExample(guid=guid, text_a=sentence_a, text_b=sentence_b, label=label))

        return examples



class MisraProcessor(object):
    def get_examples(self, data_dir, train_topic, dev_topic, test_topic, data_set):
        topics = set(['DP', 'GC', 'GM'])


        if data_set == 'test':
            filepath = os.path.join(data_dir, 'ArgPairs_' + test_topic + '.csv')
            sentences = self._read_dataset(filepath, test_topic)
            return self._get_examples(sentences, data_set)
        elif data_set=='dev':
            filepath = os.path.join(data_dir, 'ArgPairs_' + dev_topic + '.csv')
            sentences = self._read_dataset(filepath, dev_topic)
            return self._get_examples(sentences, data_set)
        else:
            all_train_examples = []

            for topic in topics:
                if topic == dev_topic or topic == test_topic:
                    continue

                filepath = os.path.join(data_dir, 'ArgPairs_' + topic + '.csv')
                sentences = self._read_dataset(filepath, topic, add_symmetry = True)
                all_train_examples.extend(self._get_examples(sentences, data_set))

            return all_train_examples


    def _read_dataset(self, filepath, topic, add_symmetry = False):
        logging.info("Read file: "+filepath)
        sentences = defaultdict(lambda: defaultdict(dict))

        with open(filepath, 'r', encoding='iso-8859-1') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            headers = next(csvreader)

            for splits in csvreader:
                assert(len(splits)==11)
                label = float(splits[0].strip())/5
                sentence_a = splits[-1].strip()
                sentence_b = splits[-2].strip()

                sentences[topic][sentence_a][sentence_b] = label

                if add_symmetry:
                    sentences[topic][sentence_b][sentence_a] = label

        return sentences



    def _get_examples(self, sentences, data_set):
        topics = list(sentences.keys())
        examples = []
        for topic in topics:
            for sentence_a in sentences[topic].keys():
                for sentence_b in sentences[topic][sentence_a].keys():
                    guid = "%s-%d" % (data_set, len(examples))
                    label = sentences[topic][sentence_a][sentence_b]
                    examples.append(InputExample(guid=guid, text_a=sentence_a, text_b=sentence_b, label=label))

        return examples


    def get_test_examples(self, data_file, data_set):
        return self.get_train_examples(data_file, data_set)






def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    tokens_a_longer_max_seq_length = 0
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        len_tokens_a = len(tokens_a)
        len_tokens_b = 0



        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            len_tokens_b = len(tokens_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        if (len_tokens_a + len_tokens_b) > (max_seq_length - 2):
            tokens_a_longer_max_seq_length += 1

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids)==max_seq_length
        assert len(input_mask)==max_seq_length
        assert len(segment_ids)==max_seq_length

        label_id = float(example.label)


        if ex_index < 1 and example.guid is not None and example.guid.startswith('train-'):
            logger.info("\n\n*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    logger.info(":: Sentences longer than max_sequence_length: %d" % (tokens_a_longer_max_seq_length))
    logger.info(":: Num sentences: %d" % (len(examples)))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(predicted_logits, gold_labels):
    assert len(predicted_logits) == len(gold_labels)

    num_labels = 0
    num_correct = 0

    for predicted_logit, gold_label in zip(predicted_logits, gold_labels):
        if gold_label < 0: #Labels < 0 indicate non-existent labels
            continue

        num_labels += 1

        #Binarize gold and predicted label
        if (gold_label < 0.5 and predicted_logit < 0.5) or (gold_label >= 0.5 and predicted_logit >= 0.5):
            num_correct += 1


    return num_correct / num_labels


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir.")
    parser.add_argument("--train_file",
                        default=None,
                        type=str)
    parser.add_argument("--dev_file",
                        default=None,
                        type=str)
    parser.add_argument("--test_file",
                        default=None,
                        type=str)


    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")




    args = parser.parse_args()



    processors = {
        "ukp_aspects": UKPAspectsProcessor,
        "misra": MisraProcessor,
    }





    if args.local_rank==-1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank!=-1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as fOut:
        fOut.write(str(args))

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = 1

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        train_examples = processor.get_examples(args.data_dir, args.train_file, args.dev_file, args.test_file, 'train')
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = SigmoidBERT.from_pretrained(args.bert_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank), num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank!=-1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank!=-1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale==0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0


    if args.do_train:

        with open(os.path.join(args.output_dir, "train_sentences.csv"), "w") as writer:
            for idx, example in enumerate(train_examples):
                writer.write("%s\t%s\t%s\n" % (example.label, example.text_a, example.text_b))


        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)
        logger.info("\n\n***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if args.local_rank==-1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)


        model.train()
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps==0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            #Dev set
            if args.dev_file is not None:
                eval_set = 'dev'
                eval_results_filename = "%s_results_epoch_%d.txt" % (eval_set, epoch+1)
                eval_prediction_filename = "%s_predictions_epoch_%d.tsv" % (eval_set, epoch+1)
                do_evaluation(processor, args, tokenizer, model, device, global_step,
                              eval_set, eval_results_filename, eval_prediction_filename)

            # Test set
            if args.test_file is not None:
                eval_set = 'test'
                eval_results_filename = "%s_results_epoch_%d.txt" % (eval_set, epoch+1)
                eval_prediction_filename = "%s_predictions_epoch_%d.tsv" % (eval_set, epoch+1)

                do_evaluation(processor, args, tokenizer, model, device, global_step,
                              eval_set, eval_results_filename, eval_prediction_filename)

    # Save a trained model
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        torch.save(model_to_save.state_dict(), output_model_file)

def do_evaluation(processor, args, tokenizer, model, device, global_step, eval_set, eval_results_filename, eval_prediction_filename):
    eval_examples = processor.get_examples(args.data_dir, args.train_file, args.dev_file, args.test_file, eval_set)
    eval_features = convert_examples_to_features(eval_examples, args.max_seq_length, tokenizer)
    logger.info("\n\n\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()

    nb_eval_steps, nb_eval_examples = 0, 0
    gold_labels = [f.label_id for f in eval_features]

    predicted_logits = []
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)


        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        predicted_logits.extend(logits[:,0])

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_accuracy = accuracy(predicted_logits, gold_labels)

    eval_spearman, eval_pearson = -999, -999

    try:
        eval_spearman, _ = scipy.stats.spearmanr(gold_labels, predicted_logits)
    except:
        pass

    try:
        eval_pearson, _ = scipy.stats.pearsonr(gold_labels, predicted_logits)
    except:
        pass

    result = {
              'eval_accuracy': eval_accuracy,
              'eval_spearman': eval_spearman,
              'eval_pearson': eval_pearson,
              'global_step': global_step,
              }

    output_eval_file = os.path.join(args.output_dir, eval_results_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("\n\n\n***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


    output_pred_file = os.path.join(args.output_dir, eval_prediction_filename)
    with open(output_pred_file, "w") as writer:
        for idx, example in enumerate(eval_examples):
            gold_label = example.label
            pred_logits = predicted_logits[idx]
            writer.write("\t".join([example.text_a.replace("\n", " ").replace("\t", " "), example.text_b.replace("\n", " ").replace("\t", " "), str(gold_label), str(pred_logits)]))
            writer.write("\n")


if __name__=="__main__":
    main()





