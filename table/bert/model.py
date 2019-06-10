import sys
from pathlib import Path
import json

import torch
from torch_scatter import scatter_max, scatter_add, scatter_mean
import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from table.bert.relation_predictor import CONFIG_NAME

NEGATIVE_NUMBER = -1e8


class BERTRelationIdentificationModel(BertPreTrainedModel):
    def __init__(self, config, output_dropout_prob=0.1, column_representation='max_pool', **kwargs):
        super(BERTRelationIdentificationModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(output_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

        self.column_representation = column_representation

    @classmethod
    def config(cls):
        return {
            'output_dropout_prob': 0.1,
            'column_representation': 'max_pooling'
        }

    @classmethod
    def build(cls, model_path):
        if isinstance(model_path, str):
            model_path = Path(model_path)

        output_config_file = model_path.parent / CONFIG_NAME
        print(f'BERT config file: {output_config_file}', file=sys.stderr)
        bert_config = BertConfig(str(output_config_file))

        config_file = model_path.parent / 'config.json'
        print(f'Model config file: {config_file}', file=sys.stderr)
        config = json.load(config_file.open())
        model = cls(bert_config, **config)

        print(f'model file: {model_path}', file=sys.stderr)
        model.load_state_dict(torch.load(str(model_path), map_location=lambda storage, location: storage))

        return model

    def get_column_representation(self,
                                  flattened_column_encoding: torch.Tensor,
                                  column_token_to_column_id: torch.Tensor,
                                  column_token_mask: torch.Tensor,
                                  column_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            flattened_column_encoding: (batch_size, total_column_token_num, encoding_size)
            column_token_to_column_id: (batch_size, total_column_token_num)
            column_mask: (batch_size, max_column_num)

        Returns:
            column_encoding: (batch_size, max_column_num, encoding_size)
        """

        method = self.column_representation
        if method == 'max_pool':
            agg_func = scatter_max
            flattened_column_encoding[column_token_mask == 0] = float('-inf')
        elif method == 'mean_pool':
            agg_func = scatter_mean

        max_column_num = column_mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size, encoding_size)
        result = agg_func(flattened_column_encoding,
                          column_token_to_column_id.unsqueeze(-1).expand(-1, -1, self.config.hidden_size),
                          dim=1,
                          dim_size=max_column_num)

        if method == 'max_pool':
            column_encoding = result[0]
        else:
            column_encoding = result

        return column_encoding

    def forward(self, input_ids, token_type_ids, attention_mask, column_token_to_column_id, column_token_mask, column_mask, labels=None, **kwargs):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)

        # grab column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        # (batch_size, max_column_size, encoding_size)
        column_encoding = self.get_column_representation(flattened_column_encoding,
                                                         column_token_to_column_id,
                                                         column_token_mask,
                                                         column_mask)

        logits = self.classifier(column_encoding)
        info = dict()

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
            return loss, info
        else:
            return logits, info


class BERTRelationIdentificationAlignmentBasedModel(BertPreTrainedModel):
    def __init__(self, config, attention_type='biaffine'):
        super(BERTRelationIdentificationAlignmentBasedModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attention_type = attention_type
        if attention_type == 'biaffine':
            self.att_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask,
                column_token_to_column_id, column_token_mask, column_mask, question_token_mask,
                labels=None, return_attention_matrix=False):
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
        att_weights_matrix = self.attention(question_token_encoding, column_encoding.permute(0, 2, 1))
        att_weights = (1. - question_token_mask.unsqueeze(-1)) * NEGATIVE_NUMBER + att_weights_matrix
        # (batch_size, max_column_num)
        att_weights, _ = torch.max(att_weights, dim=1)

        pred_info = dict()
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
            return loss, pred_info
        else:
            p = torch.sigmoid(att_weights)
            result = torch.stack([1 - p, p], dim=-1)
            if return_attention_matrix:
                pred_info['attention_matrix'] = att_weights_matrix

            return result, pred_info

    def attention(self, key, value):
        if self.attention_type == 'biaffine':
            key = self.att_linear(key)

        return torch.bmm(key, value)
