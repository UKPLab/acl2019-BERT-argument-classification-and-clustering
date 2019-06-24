import torch
from torch import nn
import pytorch_pretrained_bert.modeling


class SigmoidBERT(pytorch_pretrained_bert.modeling.BertPreTrainedModel):
    def __init__(self, config, num_labels=1):
        super(SigmoidBERT, self).__init__(config)
        assert num_labels==1;
        self.num_labels = num_labels
        self.bert = pytorch_pretrained_bert.modeling.BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.lin_layer = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        #sent_encoding = pooled_output
        sent_encoding = encoded_layers[:, 0, :]

        sent_encoding = self.lin_layer(sent_encoding)
        logits = torch.sigmoid(sent_encoding)

        if labels is not None:
            loss_fct = nn.BCELoss()
            loss = loss_fct(logits[:, 0], labels.view(-1))
            return loss
        else:
            return logits