import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch import nn as nn

from nsm.parser_module.bert_encoder import BertEncoder
from nsm.parser_module.encoder import EncoderBase, ContextEncoding, COLUMN_TYPES
#from table.bert.data_model import Example
#from table.bert.model import TableBERT, ContentEncodingTableBERT, TableBertConfig
from table_bert.content_table_bert import ContentBasedTableBert


class ContentBasedEncoder(BertEncoder):
    def __init__(
        self,
        table_bert_model,
        output_size: int,
        question_feat_size: int,
        builtin_func_num: int,
        memory_size: int,
        column_feature_num: int,
        max_row_num: int,
        dropout: float,
        config: Dict,
    ):
        assert config['table_representation'] == 'canonical'

        super(ContentBasedEncoder, self).__init__(
            table_bert_model,
            output_size=output_size,
            question_feat_size=question_feat_size,
            column_feature_num=column_feature_num,
            builtin_func_num=builtin_func_num,
            memory_size=memory_size,
            config=config,
            dropout=dropout
        )

        self.max_row_num = max_row_num

        self.init_weights()

    @property
    def sufficient_context_encoding_entries(self):
        keys = ['question_encoding', 'question_mask', 'question_encoding_att_linear',
                'table_encoding', 'table_mask', 'column_mask']
        return keys

    @classmethod
    def build(cls, config, table_bert_model=None):
        if table_bert_model is None:
            table_bert_model = cls.get_table_bert_model(config, ContentBasedTableBert)

        return cls(
            table_bert_model,
            output_size=config['hidden_size'],
            max_row_num=3,
            question_feat_size=config['n_en_input_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            column_feature_num=config['n_de_output_features'],
            dropout=config['dropout'],
            config=config
        )

    def bert_encode(self, env_context: List[Dict]) -> Any:
        question_encoding, table_encoding, info = self.bert_model.encode(
            [e['question_tokens'] for e in env_context],
            [e['table'].with_rows(e['table'].data[:self.max_row_num]) for e in env_context]
            # [Example(
            #     question=e['question_tokens'],
            #     table=e['table'].with_rows(e['table'].data[:self.max_row_num])
            # )
            # for e in env_context]
        )

        return question_encoding, table_encoding, info['tensor_dict']

    def encode(self, env_context: List[Dict]) -> ContextEncoding:
        batched_context = self.example_list_to_batch(env_context)

        question_encoding_dict, table_encoding_dict, encoding_info = self.bert_encode(env_context)

        # remove leading [CLS] symbol
        question_encoding = question_encoding_dict['value'][:, 1:]
        question_mask = question_encoding_dict['mask'][:, 1:]
        cls_encoding = question_encoding_dict['value'][:, 0]

        table_encoding = table_encoding_dict['value']

        if self.question_feat_size > 0:
            question_encoding = torch.cat([
                question_encoding,
                batched_context['question_features']],
                dim=-1)

        question_encoding = self.bert_output_project(question_encoding)
        question_encoding_att_linear = self.question_encoding_att_value_to_key(question_encoding)

        batch_size = len(env_context)
        constant_value_num = batched_context['constant_spans'].size(1)
        max_raw_column_num = max(
            len(ctx['table'].column_info['raw_columns'])
            for ctx in env_context
        )

        new_tensor = table_encoding.new_tensor

        raw_column_canonical_ids = np.zeros((batch_size, max_raw_column_num), dtype=np.int64)
        column_type_ids = np.zeros((batch_size, max_raw_column_num), dtype=np.int64)
        raw_column_mask = np.zeros((batch_size, max_raw_column_num), dtype=np.float32)

        for e_id, context in enumerate(env_context):
            column_info = context['table'].column_info
            raw_columns = column_info['raw_columns']
            raw_column_canonical_ids[e_id, :len(raw_columns)] = column_info['raw_column_canonical_ids']
            column_type_ids[e_id, :len(raw_columns)] = [
                self.column_type_to_id[col.type] for col in raw_columns]

            raw_column_mask[e_id, :len(raw_columns)] = 1.

        raw_column_canonical_ids = new_tensor(raw_column_canonical_ids, dtype=torch.long)
        row_num, column_encoding_sie = table_encoding.size(1), table_encoding.size(-1)
        exp_raw_column_canonical_ids = raw_column_canonical_ids[:, None, :, None].expand(
            -1, row_num, -1, column_encoding_sie)

        table_encoding = torch.gather(
            table_encoding,
            dim=2,  # column_num
            index=exp_raw_column_canonical_ids
        )

        if self.config['use_column_type_embedding']:
            type_fused_column_encoding = torch.cat([
                table_encoding,
                self.column_type_embedding(new_tensor(column_type_ids, dtype=torch.long)).unsqueeze(1).expand(-1, table_encoding.size(1), -1, -1)  # add row dimension
            ], dim=-1)

            table_encoding = type_fused_column_encoding

        raw_column_mask = new_tensor(raw_column_mask)
        row_mask, _ = table_encoding_dict['mask'].max(dim=-1, keepdim=True)
        row_expanded_column_mask = raw_column_mask.unsqueeze(1).expand(-1, table_encoding.size(1), -1)
        raw_table_mask = row_mask * row_expanded_column_mask

        table_mask = raw_table_mask
        column_mask = raw_column_mask
        table_encoding = table_encoding * table_mask.unsqueeze(-1)

        # trim the encoding to fit the executor's memory
        if max_raw_column_num > constant_value_num:
            table_encoding = table_encoding[:, :, :constant_value_num, :]
            table_mask = table_mask[:, :, :constant_value_num]
            column_mask = raw_column_mask[:, :constant_value_num]

        # (batch_size, max_column_num, encoding_size)
        table_encoding_proj = self.bert_table_output_project(table_encoding)
        # initial
        constant_value_embedding = table_encoding.new_zeros(batch_size, constant_value_num, table_encoding_proj.size(-1))

        constant_encoding, constant_mask = self.get_constant_encoding(
            question_encoding, batched_context['constant_spans'], constant_value_embedding,
            torch.cat([
                column_mask,
                column_mask.new_zeros(batch_size, constant_value_num - column_mask.size(1))
            ], dim=-1)
        )

        context_encoding = {
            'batch_size': len(env_context),
            'question_encoding': question_encoding,
            'question_mask': question_mask,
            'question_encoding_att_linear': question_encoding_att_linear,
            'table_encoding': table_encoding_proj,
            'table_mask': table_mask,
            'column_mask': column_mask,
            'cls_encoding': cls_encoding,
            'constant_encoding': constant_encoding,
            'constant_mask': constant_mask
        }

        return context_encoding
