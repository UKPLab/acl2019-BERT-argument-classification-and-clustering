"""
This code runs a fine-tuned BERT model to estimate the similarity between two arguments.

In this example, we include arguments from three topics (courtesy to www.procon.org): Zoos, Vegetarianism, Climate Change

Each argument is compared against each other argument. Arguments from the same topic, e.g. on zoos, should be ranked higher
than arguments from different topics.

Usage: python inference.py
"""

from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from train import InputExample, convert_examples_to_features
from SigmoidBERT import SigmoidBERT

# See the README.md where to download pre-trained models
#model_path = 'bert_output/ukp_aspects_all' #ukp_aspects_all model: trained
model_path = 'bert_output/misra_all' #misra_all model: Trained on all 3 topics from Misra et al., 2016


max_seq_length = 64
eval_batch_size = 8

arguments = ['Zoos save species from extinction and other dangers.',
             'Zoos produce helpful scientific research.',
             'Zoos are detrimental to animals\' physical health.',
             'Zoo confinement is psychologically damaging to animals.',
             'Eating meat is not cruel or unethical; it is a natural part of the cycle of life. ',
             'It is cruel and unethical to kill animals for food when vegetarian options are available',
             'Overwhelming scientific consensus says human activity is primarily responsible for global climate change.',
             'Rising levels of human-produced gases released into the atmosphere create a greenhouse effect that traps heat and causes global warming.'
             ]

#Compare every argument with each other
input_examples = []
output_examples = []

for i in range(0, len(arguments)-1):
    for j in range(i+1, len(arguments)):
        input_examples.append(InputExample(text_a=arguments[i], text_b=arguments[j], label=-1))
        output_examples.append([arguments[i], arguments[j]])


tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
eval_features = convert_examples_to_features(input_examples, max_seq_length, tokenizer)

all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SigmoidBERT.from_pretrained(model_path,)
model.to(device)
model.eval()

predicted_logits = []
with torch.no_grad():
    for input_ids, input_mask, segment_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        logits = model(input_ids, segment_ids, input_mask)
        logits = logits.detach().cpu().numpy()
        predicted_logits.extend(logits[:, 0])




for idx in range(len(predicted_logits)):
    output_examples[idx].append(predicted_logits[idx])

#Sort by similarity
output_examples = sorted(output_examples, key=lambda x: x[2], reverse=True)

print("Predicted similarities (sorted by similarity):")
for idx in range(len(output_examples)):
    example = output_examples[idx]
    print("Sentence A:", example[0])
    print("Sentence B:", example[1])
    print("Similarity:", example[2])
    print("")

