import sys
import json
from pathlib import Path
from typing import Dict

from table_bert.config import TableBertConfig, BERT_CONFIGS
from table_bert.table_bert import TableBertModel
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert
from table_bert.vanilla_table_bert import VanillaTableBert


def get_table_bert_model(config: Dict, use_proxy=False, master=None):
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

    if use_proxy:
        from nsm.parser_module.table_bert_proxy import TableBertProxy
        tb_config = TableBertConfig.from_file(tb_config_file)
        table_bert_model = TableBertProxy(actor_id=master, table_bert_config=tb_config)
    else:
        table_bert_extra_config = config.get('table_bert_extra_config', dict())
        table_bert_model = table_bert_cls.load(
            tb_path,
            tb_config_file,
            **table_bert_extra_config
        )

        if type(table_bert_model) == VanillaTableBert:
            table_bert_model.config.column_representation = config.get('column_representation', 'mean_pool')

        print('Table Bert Config', file=sys.stderr)
        print(json.dumps(vars(table_bert_model.config), indent=2), file=sys.stderr)

    return table_bert_model
