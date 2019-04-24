import torch
from torch_scatter import scatter_max, scatter_add
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

NEGATIVE_NUMBER = -1e8


class BERTRelationIdentificationModel(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super(BERTRelationIdentificationModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, column_token_to_column_id, column_token_mask, column_mask, labels=None, **kwargs):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)

        # grab column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        flattened_column_encoding[column_token_mask == 0] = float('-inf')
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size, encoding_size)
        column_encoding, _ = scatter_max(flattened_column_encoding,
                                         column_token_to_column_id.unsqueeze(-1).expand(-1, -1, self.config.hidden_size),
                                         dim=1, dim_size=column_mask.size(-1))

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


class BERTRelationIdentificationAlignmentBasedModel(BertPreTrainedModel):
    def __init__(self, config, attention_type='biaffine'):
        super(BERTRelationIdentificationAlignmentBasedModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attention_type = attention_type
        if attention_type == 'biaffine':
            self.att_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, column_token_to_column_id, column_token_mask, column_mask, question_token_mask, labels=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)

        # (batch_size, max_question_len, encoding_size)
        max_question_len = question_token_mask.size(-1)
        question_token_encoding = sequence_output[:, :max_question_len].clone()

        # grab column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        flattened_column_encoding[column_token_mask == 0] = float('-inf')
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size, encoding_size)
        column_encoding, _ = scatter_max(flattened_column_encoding,
                                         column_token_to_column_id.unsqueeze(-1).expand(-1, -1, self.config.hidden_size),
                                         dim=1, dim_size=column_mask.size(-1))

        # (batch_size, max_question_len, max_column_num)
        att_weights = self.attention(question_token_encoding, column_encoding.permute(0, 2, 1))
        att_weights = (1. - question_token_mask.unsqueeze(-1)) * NEGATIVE_NUMBER + att_weights
        # (batch_size, max_column_num)
        att_weights, _ = torch.max(att_weights, dim=1)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if column_mask is not None:
                active_loss = column_mask.view(-1) == 1
                active_logits = att_weights.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels.float())
            else:
                loss = loss_fct(att_weights.view(-1), labels.view(-1))
            return loss
        else:
            p = torch.sigmoid(att_weights)
            result = torch.stack([1 - p, p], dim=-1)

            return result

    def attention(self, key, value):
        if self.attention_type == 'biaffine':
            key = self.att_linear(key)

        return torch.bmm(key, value)
