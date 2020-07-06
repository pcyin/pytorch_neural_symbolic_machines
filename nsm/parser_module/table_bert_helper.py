from typing import Dict, List, Tuple, Any
import copy
import sys
import json
import numpy as np
from pathlib import Path

from table_bert.config import TableBertConfig, BERT_CONFIGS
from table_bert.table_bert import TableBertModel
from table_bert.vertical.config import VerticalAttentionTableBertConfig
from table_bert.table import Column, Table
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert
from table_bert.vanilla_table_bert import VanillaTableBert

from nsm.parser_module.sequence_util import StringMatchUtil


def get_table_bert_model(config: Dict, use_proxy=False, master=None):
    model_name_or_path = config.get('table_bert_model_or_config')
    if model_name_or_path in {None, ''}:
        model_name_or_path = config.get('table_bert_config_file')
    if model_name_or_path in {None, ''}:
        model_name_or_path = config.get('table_bert_model')

    table_bert_extra_config = config.get('table_bert_extra_config', dict())

    print(f'Loading table BERT model {model_name_or_path}', file=sys.stderr)
    model = TableBertModel.from_pretrained(
        model_name_or_path,
        **table_bert_extra_config
    )

    if type(model) == VanillaTableBert:
        model.config.column_representation = config.get('column_representation', 'mean_pool_column_name')

    if use_proxy:
        from nsm.parser_module.table_bert_proxy import TableBertProxy
        tb_config = copy.deepcopy(model.config)
        del model
        model = TableBertProxy(actor_id=master, table_bert_config=tb_config)

    print('Table Bert Config', file=sys.stderr)
    print(json.dumps(vars(model.config), indent=2), file=sys.stderr)

    return model


def get_table_bert_model_deprecated(config: Dict, use_proxy=False, master=None):
    tb_path = config.get('table_bert_model_or_config')
    if tb_path is None or tb_path == '':
        tb_path = config.get('table_bert_config_file')
    if tb_path is None or tb_path == '':
        tb_path = config.get('table_bert_model')

    tb_path = Path(tb_path)
    assert tb_path.exists()

    if tb_path.suffix == '.json':
        tb_config_file = tb_path
        tb_path = None
    else:
        print(f'Loading table BERT model {tb_path}', file=sys.stderr)
        tb_config_file = tb_path.parent / 'tb_config.json'

    if use_proxy:
        from nsm.parser_module.table_bert_proxy import TableBertProxy
        tb_config = TableBertConfig.from_file(tb_config_file)
        table_bert_model = TableBertProxy(actor_id=master, table_bert_config=tb_config)
    else:
        table_bert_extra_config = config.get('table_bert_extra_config', dict())
        # if it is a not pre-trained model, we use the default parameters
        if tb_path is None:
            table_bert_cls = TableBertConfig.infer_model_class_from_config_file(tb_config_file)
            print(f'Creating a default {table_bert_cls.__name__} without pre-trained parameters!', file=sys.stderr)

            table_bert_model = table_bert_cls(
                config=table_bert_cls.CONFIG_CLASS.from_file(
                    tb_config_file, **table_bert_extra_config
                )
            )
        else:
            table_bert_model = TableBertModel.from_pretrained(
                tb_path,
                **table_bert_extra_config
            )

        if type(table_bert_model) == VanillaTableBert:
            table_bert_model.config.column_representation = config.get('column_representation', 'mean_pool_column_name')

        print('Table Bert Config', file=sys.stderr)
        print(json.dumps(vars(table_bert_model.config), indent=2), file=sys.stderr)

    return table_bert_model


def model_use_vertical_attention(bert_model):
    return isinstance(bert_model.config, VerticalAttentionTableBertConfig)


def get_question_biased_sampled_rows(context, table, num_rows=3):
        candidate_row_match_score = {}
        for row_id, row in enumerate(table.data):
            row_data = list(row.values() if isinstance(row, dict) else row)
            for cell in row_data:
                if len(cell) > 0 and StringMatchUtil.contains(context, cell) and not StringMatchUtil.all_stop_words(cell):
                    candidate_row_match_score[row_id] = max(
                        candidate_row_match_score.get(row_id, 0),
                        len(cell)
                    )

        candidate_row_ids = [idx for idx, score in candidate_row_match_score.items() if score > 0]
        if len(candidate_row_ids) < num_rows:
            # find partial match
            max_ngram_num = 3
            for row_id, row in enumerate(table.data):
                if row_id in candidate_row_ids:
                    continue

                row_data = list(row.values() if isinstance(row, dict) else row)

                for cell in row_data:
                    found = False
                    if len(cell) > 0:
                        for ngram_num in reversed(range(1, max_ngram_num + 1)):
                            for start_idx in range(0, len(cell) - ngram_num + 1):
                                end_idx = start_idx + ngram_num
                                ngram = cell[start_idx: end_idx]
                                if not StringMatchUtil.all_stop_words(ngram) and StringMatchUtil.contains(context, ngram):
                                    candidate_row_match_score[row_id] = max(
                                        ngram_num,
                                        candidate_row_match_score.get(row_id, 0)
                                    )
                                    found = True

                                if found: break
                            if found: break

            candidate_row_ids = [idx for idx, score in candidate_row_match_score.items() if score > 0]
            if len(candidate_row_ids) < num_rows:
                not_included_row_ids = [idx for idx in range(len(table)) if idx not in candidate_row_ids]
                left = num_rows - len(candidate_row_ids)
                for idx in not_included_row_ids[:left]: candidate_row_match_score[idx] = 0
                candidate_row_ids = candidate_row_ids + not_included_row_ids[:left]

        top_k_row_ids_by_match_score = sorted(
            candidate_row_ids,
            key=lambda row_id: -candidate_row_match_score[row_id])[:num_rows]
        top_k_row_ids_by_match_score = sorted(top_k_row_ids_by_match_score)
        candidate_rows = [table.data[idx] for idx in top_k_row_ids_by_match_score]

        return candidate_rows


def get_question_biased_sampled_cells(context, table):
        candidate_cells = [[] for column in table.header]

        for col_idx, column in enumerate(table.header):
            cell_match_scores = []

            for row in table.data:
                cell = row.get(table.header[col_idx].name, []) if isinstance(row, dict) else row[col_idx]
                if len(cell) > 0 and StringMatchUtil.contains(context, cell) and not StringMatchUtil.all_stop_words(cell):
                    cell_match_scores.append((cell, len(cell)))

            if len(cell_match_scores) == 0:
                # use partial match
                max_ngram_num = 3

                for row_id, row in enumerate(table.data):
                    cell = row.get(table.header[col_idx].name, []) if isinstance(row, dict) else row[col_idx]
                    found = False
                    if len(cell) > 0:
                        for ngram_num in reversed(range(1, max_ngram_num + 1)):
                            for start_idx in range(0, len(cell) - ngram_num + 1):
                                end_idx = start_idx + ngram_num
                                ngram = cell[start_idx: end_idx]
                                if not StringMatchUtil.all_stop_words(ngram) and StringMatchUtil.contains(context, ngram):
                                    cell_match_scores.append((cell, ngram_num))
                                    found = True

                                if found: break

                            if found: break

            best_matched_cell = sorted(cell_match_scores, key=lambda x: -x[1])
            if best_matched_cell:
                best_matched_cell = best_matched_cell[0][0]
            else:
                best_matched_cell = column.sample_value_tokens

            candidate_cells[col_idx] = best_matched_cell

        return candidate_cells


def get_table_bert_input_from_context(
    env_context: List[Dict],
    bert_model: TableBertModel,
    is_training: bool,
    **kwargs
) -> Tuple[List[Any], List[Table]]:
    contexts = []
    tables = []

    content_snapshot_strategy = kwargs.get('content_snapshot_strategy', None)
    if content_snapshot_strategy:
        assert content_snapshot_strategy in ('sampled_rows', 'synthetic_row')

    for e in env_context:
        contexts.append(e['question_tokens'])

        if model_use_vertical_attention(bert_model):
            sample_row_num = bert_model.config.sample_row_num
            if content_snapshot_strategy == 'sampled_rows':
                if 'sampled_rows' not in e:
                    sampled_rows = get_question_biased_sampled_rows(
                        e['question_tokens'], e['table'],
                        num_rows=sample_row_num
                    )
                    e['sampled_rows'] = sampled_rows

                sampled_rows = e['sampled_rows']
            else:
                if is_training:
                    sampled_rows = [
                        e['table'].data[idx]
                        for idx
                        in sorted(
                            np.random.choice(
                                list(range(len(e['table']))),
                                replace=False,
                                size=sample_row_num
                            )
                        )
                    ]
                else:
                    sampled_rows = e['table'].data[:sample_row_num]

            table = e['table'].with_rows(sampled_rows)
        else:
            table = e['table']
            if content_snapshot_strategy:
                if 'sampled_rows' not in e:
                    if content_snapshot_strategy == 'sampled_rows':
                        sampled_rows = get_question_biased_sampled_rows(
                            e['question_tokens'], e['table'],
                            num_rows=1
                        )
                        e['sampled_rows'] = sampled_rows
                    elif content_snapshot_strategy == 'synthetic_row':
                        sampled_cells = get_question_biased_sampled_cells(
                            e['question_tokens'], e['table']
                        )
                        e['sampled_rows'] = [sampled_cells]

                sampled_row = e['sampled_rows'][0]
                new_header = []
                for idx, column in enumerate(e['table'].header):
                    cell_value = sampled_row[idx] if isinstance(sampled_row, list) else sampled_row[column.name]
                    new_column = Column(
                        name=column.name, name_tokens=column.name_tokens, type=column.type,
                        sample_value=cell_value, sample_value_tokens=cell_value
                    )
                    new_header.append(new_column)

                table = Table(
                    id=table.id, header=new_header,
                    data=[{column.name: column.sample_value_tokens for column in new_header}]
                )

        tables.append(table)

    return contexts, tables
