from typing import Dict

from nsm import data_utils
from nsm.execution.executor_factory import TableExecutor
from nsm.computer_factory import SPECIAL_TKS as INTERPRETER_SPECIAL_TOKENS


class WikiTableExecutor(TableExecutor):
    pass


def init_world_config() -> Dict:
    dummy_kg = {
        'kg': None,
        'num_props': [],
        'datetime_props': [],
        'props': [],
        'row_ents': []
    }

    api = WikiTableExecutor(dummy_kg).get_api()
    op_vocab = data_utils.Vocab(
        [f['name'] for f in api['func_dict'].values()] +
        ['all_rows'] +
        INTERPRETER_SPECIAL_TOKENS
    )

    config = {
        'interpreter_builtin_func_num': op_vocab.size,
        'executor_api': api
    }

    return config


world_config = init_world_config()
