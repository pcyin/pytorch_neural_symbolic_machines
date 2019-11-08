import json
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer
from table_bert.table_bert import TableBertModel
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert
from torch import nn as nn

from nsm.parser_module.encoder import EncoderBase, ContextEncoding, COLUMN_TYPES

from table_bert.vanilla_table_bert import VanillaTableBert
# from table.bert.data_model import Example
# from table.bert.model import TableBERT

Example = namedtuple('Example', ['question', 'table'])


class BertEncoder(EncoderBase):
    def __init__(
        self,
        table_bert_model,
        output_size: int,
        config: Dict,
        question_feat_size: int,
        builtin_func_num: int,
        memory_size: int,
        column_feature_num: int,
        dropout: float = 0.
    ):
        EncoderBase.__init__(self, output_size, builtin_func_num, memory_size)

        self.config = config
        self.bert_model = table_bert_model
        self.question_feat_size = question_feat_size
        self.dropout = nn.Dropout(dropout)
        self.max_variable_num_on_memory = memory_size - builtin_func_num
        self.column_feature_num = column_feature_num

        self.bert_output_project = nn.Linear(
            self.bert_model.output_size + self.question_feat_size,
            self.output_size, bias=False
        )

        self.question_encoding_att_value_to_key = nn.Linear(
            self.output_size,
            self.output_size, bias=False
        )

        if self.config['table_representation'] == 'canonical':
            self.column_type_to_id = {t: i for i, t in enumerate(COLUMN_TYPES)}
            self.column_type_embedding = nn.Embedding(len(self.column_type_to_id), self.config['value_embedding_size'])

        if self.config['use_column_type_embedding']:
            self.bert_table_output_project = nn.Linear(
                self.bert_model.output_size + self.column_type_embedding.embedding_dim,
                self.output_size,
                bias=False
            )
        else:
            self.bert_table_output_project = nn.Linear(
                self.bert_model.output_size, self.output_size, bias=False
            )

        self.constant_value_embedding_linear = lambda x: x

        self.init_weights()

    def init_weights(self):
        def _init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.bert_model.bert_config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        modules = [
            module
            for name, module
            in self._modules.items()
            if module and 'bert_model' not in name
        ]

        for module in modules:
            module.apply(_init_weights)

    @staticmethod
    def get_table_bert_model(config):
        tb_path = config.get('table_bert_model')

        if tb_path in ('vertical', 'vanilla'):
            tb_config_file = config['table_bert_config_file']
            table_bert_cls = {
                'vertical': VerticalAttentionTableBert,
                'vanilla': VanillaTableBert
            }[tb_path]
            tb_path = None
        else:
            print(f'Loading table BERT model {tb_path}', file=sys.stderr)
            tb_path = Path(tb_path)
            tb_config_file = tb_path.parent / 'tb_config.json'
            table_bert_cls = TableBertModel

        table_bert_model = table_bert_cls.load(
            tb_path,
            tb_config_file,
        )

        if type(table_bert_model) == VanillaTableBert:
            table_bert_model.config.column_representation = config.get('column_representation', 'mean_pool')

        print('Table Bert Config', file=sys.stderr)
        print(json.dumps(vars(table_bert_model.config), indent=2), file=sys.stderr)

        return table_bert_model

    @classmethod
    def build(cls, config, table_bert_model=None):
        if table_bert_model is None:
            table_bert_model = cls.get_table_bert_model(config)

        return cls(
            table_bert_model,
            output_size=config['hidden_size'],
            question_feat_size=config['n_en_input_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            column_feature_num=config['n_de_output_features'],
            dropout=config['dropout'],
            config=config
        )

    def example_list_to_batch(self, env_context: List[Dict]) -> Dict:
        # self.context = dict(question_word_ids=en_inputs,
        #                                 constant_spans=constant_spans,
        #                                 question_features=question_annotation['features'],
        #                                 question_tokens=tokens,
        #                                 question_sub_tokens=tokens)

        batch_dict = dict()
        for key in ('constant_spans', 'question_features'):
            val_list = [x[key] for x in env_context]

            # (batch_size, max_entry_num, entry_dim)
            if key == 'question_features':
                max_entry_num = max(len(val) for val in val_list)
                dtype = np.float32
            else:
                max_entry_num = self.max_variable_num_on_memory
                dtype = np.int64

            entry_dim = len(val_list[0][0])
            batch_size = len(env_context)

            batch_value_tensor = np.zeros((batch_size, max_entry_num, entry_dim), dtype=dtype)

            if key == 'constant_spans':
                batch_value_tensor.fill(-1.)

            for i, val in enumerate(val_list):
                entry_num = len(val)
                batch_value_tensor[i, :entry_num] = val

            batch_dict[key] = torch.from_numpy(batch_value_tensor).to(next(self.parameters()).device)

        return batch_dict

    def _bert_encode(self, env_context: List[Dict]) -> Any:
        question_encoding, table_column_encoding, info = self.bert_model.encode(
            contexts=[
                e['question_tokens']
                for e in env_context
            ],
            tables=[
                e['table'].with_rows(e['table'].data[:self.bert_model.config.sample_row_num])
                for e in env_context
            ]
        )

        # table_bert_encoding = {
        #     'question_encoding': question_encoding['value'],
        #     'column_encoding': table_column_encoding['value'],
        # }

        table_bert_encoding = {
            'question_encoding': question_encoding,
            'column_encoding': table_column_encoding
        }

        table_bert_encoding.update(info['tensor_dict'])
        # table_bert_encoding['context_token_mask'] = question_encoding['mask']
        # table_bert_encoding['column_mask'] = table_column_encoding['mask']

        return table_bert_encoding

    def encode(self, env_context: List[Dict]) -> ContextEncoding:
        batched_context = self.example_list_to_batch(env_context)

        table_bert_encoding = self._bert_encode(env_context)

        # remove leading [CLS] symbol
        question_encoding = table_bert_encoding['question_encoding'][:, 1:]
        question_mask = table_bert_encoding['context_token_mask'][:, 1:]
        cls_encoding = table_bert_encoding['question_encoding'][:, 0]

        canonical_column_encoding = table_column_encoding = table_bert_encoding['column_encoding']
        canonical_column_mask = table_column_mask = table_bert_encoding['column_mask']

        if self.question_feat_size > 0:
            question_encoding = torch.cat([
                question_encoding,
                batched_context['question_features']],
                dim=-1)

        question_encoding = self.bert_output_project(question_encoding)

        question_encoding_att_linear = self.question_encoding_att_value_to_key(question_encoding)

        batch_size = len(env_context)
        max_column_num = table_column_encoding.size(1)
        constant_value_num = batched_context['constant_spans'].size(1)

        if self.config['table_representation'] == 'canonical':
            new_tensor = table_column_encoding.new_tensor
            canonical_column_encoding = table_column_encoding

            raw_column_canonical_ids = np.zeros((batch_size, constant_value_num), dtype=np.int64)
            column_type_ids = np.zeros((batch_size, constant_value_num), dtype=np.int64)
            raw_column_mask = np.zeros((batch_size, constant_value_num), dtype=np.float32)

            for e_id, context in enumerate(env_context):
                column_info = context['table'].column_info
                raw_columns = column_info['raw_columns']
                valid_column_num = min(constant_value_num, len(raw_columns))
                raw_column_canonical_ids[e_id, :valid_column_num] = column_info['raw_column_canonical_ids'][:valid_column_num]
                column_type_ids[e_id, :valid_column_num] = [
                    self.column_type_to_id[col.type] for col in raw_columns][:valid_column_num]

                raw_column_mask[e_id, :valid_column_num] = 1.

            raw_column_canonical_ids = new_tensor(raw_column_canonical_ids, dtype=torch.long)
            table_column_encoding = torch.gather(
                canonical_column_encoding,
                dim=1,
                index=raw_column_canonical_ids.unsqueeze(-1).expand(-1, -1, table_column_encoding.size(-1))
            )

            if self.config['use_column_type_embedding']:
                type_fused_column_encoding = torch.cat([
                    table_column_encoding, self.column_type_embedding(new_tensor(column_type_ids, dtype=torch.long))
                ], dim=-1)

                table_column_encoding = type_fused_column_encoding

            canonical_column_mask = table_column_mask
            table_column_mask = new_tensor(raw_column_mask)
            table_column_encoding = table_column_encoding * table_column_mask.unsqueeze(-1)
            max_column_num = table_column_encoding.size(1)

        # (batch_size, max_column_num, encoding_size)
        table_column_encoding = self.bert_table_output_project(table_column_encoding)

        if max_column_num < constant_value_num:
            constant_value_embedding = torch.cat([
                table_column_encoding,
                table_column_encoding.new_zeros(
                    batch_size, constant_value_num - max_column_num, table_column_encoding.size(-1))],
                dim=1)
        else:
            constant_value_embedding = table_column_encoding[:, :constant_value_num, :]

        constant_encoding, constant_mask = self.get_constant_encoding(
            question_encoding, batched_context['constant_spans'], constant_value_embedding, table_column_mask)

        context_encoding = {
            'batch_size': len(env_context),
            'question_encoding': question_encoding,
            'question_mask': question_mask,
            'question_encoding_att_linear': question_encoding_att_linear,
            'column_encoding': table_column_encoding,
            'column_mask': table_column_mask,
            'canonical_column_encoding': canonical_column_encoding,
            'canonical_column_mask': canonical_column_mask,
            'cls_encoding': cls_encoding,
            'table_bert_encoding': table_bert_encoding,
            'constant_encoding': constant_encoding,
            'constant_mask': constant_mask
        }

        return context_encoding

    def get_constant_encoding(self, question_token_encoding, constant_span, constant_value_embedding, column_mask):
        """
        Args:
            question_token_encoding: (batch_size, max_question_len, encoding_size)
            constant_span: (batch_size, mem_size, 2)
            constant_value_embedding: (batch_size, constant_value_num, embed_size)
            column_mask: (batch_size, constant_value_num)
        """
        # (batch_size, mem_size)
        constant_span_mask = torch.ge(constant_span, 0)[:, :, 0].float()

        # mask out entries <= 0
        constant_span = constant_span * constant_span_mask.unsqueeze(-1).long()

        constant_span_size = constant_span.size()
        mem_size = constant_span_size[1]
        batch_size = question_token_encoding.size(0)

        # (batch_size, mem_size, 2, embed_size)
        constant_span_embedding = torch.gather(
            question_token_encoding.unsqueeze(1).expand(-1, mem_size, -1, -1),
            index=constant_span.unsqueeze(-1).expand(-1, -1, -1, question_token_encoding.size(-1)),
            dim=2  # over `max_question_len`
        )

        # (batch_size, mem_size, embed_size)
        # constant_span_embedding = self._question_token_span_to_memory_embedding(constant_span_embedding)
        constant_span_embedding = torch.mean(constant_span_embedding, dim=-2)
        constant_span_embedding = constant_span_embedding * constant_span_mask.unsqueeze(-1)

        # `constant_value_embedding` consists mostly of table header embedding computed by table BERT
        # (batch_size, constant_value_num, embed_size)
        constant_value_embedding = self.constant_value_embedding_linear(constant_value_embedding)

        constant_encoding = constant_value_embedding + constant_span_embedding
        constant_mask = (constant_span_mask.byte() | column_mask.byte()).float()

        return constant_encoding, constant_mask
