import torch
from torch_scatter import scatter_max, scatter_add
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss


class BERTRelationIdentificationModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BERTRelationIdentificationModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, column_token_to_column_id, column_token_mask, column_mask, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)

        # grab column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        flattened_column_encoding[column_token_mask == 0] = float('-inf')
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size, encoding_size)
        column_encoding, _ = scatter_max(flattened_column_encoding, column_token_to_column_id.unsqueeze(-1).expand(-1, -1, self.config.hidden_size), dim=1)

        logits = self.classifier(column_encoding)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if column_mask is not None:
                active_loss = column_mask.view(-1) == 1
                active_logits = logits.view(-1, 2)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            return loss
        else:
            return logits
