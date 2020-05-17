"""
Pytorch implementation of neural symbolic machines

Usage:
    experiments.py train --work-dir=<dir> --config=<file> [options]
    experiments.py test --model=<file> --test-file=<file> [options]

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --work-dir=<dir>                        work directory
    --config=<file>                         path to config file
    --extra-config=<str>                    Extra configuration [default: {}]
    --seed=<int>                            seed [default: 0]
    --eval-batch-size=<int>                 batch size for evaluation [default: 32]
    --eval-beam-size=<int>                  beam size for evaluation [default: 5]
    --save-decode-to=<file>                 save decoding results to file [default: None]
"""

import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Iterable, Any, Optional, Union
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer

import nsm.execution.worlds.wikisql
import nsm.execution.worlds.wikitablequestions
from nsm.actor import Actor
from nsm.parser_module.agent import PGAgent
from nsm.embedding import EmbeddingModel
from nsm.env_factory import QAProgrammingEnv
from nsm.computer_factory import LispInterpreter
from nsm.data_utils import Vocab
import nsm.execution.executor_factory as executor_factory
import table.utils as utils
from nsm.data_utils import load_jsonl
from nsm.evaluator import Evaluator, Evaluation
from nsm.learner import Learner

import multiprocessing

from docopt import docopt

from nsm.program_cache import SharedProgramCache
# from table.bert.data_model import Column
from table_bert.dataset import Column, Table


def annotate_example_for_bert(
    example: Dict, table: Dict,
    bert_tokenizer: BertTokenizer,
    table_representation_method: Optional[str] = 'canonical'
):
    e_id = example['id']

    # sub-tokenize the question
    question_tokens = example['tokens']
    example['original_tokens'] = question_tokens
    token_position_map = OrderedDict()   # map of token index before and after sub-tokenization

    question_feature = example['features']

    cur_idx = 0
    new_question_feature = []
    question_subtokens = []
    for old_idx, token in enumerate(question_tokens):
        if token == '<DECODE>': token = '[MASK]'
        if token == '<START>': token = '[MASK]'

        sub_tokens = bert_tokenizer.tokenize(token)
        question_subtokens.extend(sub_tokens)

        token_new_idx_start = cur_idx
        token_new_idx_end = cur_idx + len(sub_tokens)
        token_position_map[old_idx] = (token_new_idx_start, token_new_idx_end)
        new_question_feature.extend([question_feature[old_idx]] * len(sub_tokens))

        cur_idx = token_new_idx_end

    token_position_map[len(question_tokens)] = (len(question_subtokens), len(question_subtokens))

    example['tokens'] = question_subtokens
    example['features'] = new_question_feature

    for entity in example['entities']:
        old_token_start = entity['token_start']
        old_token_end = entity['token_end']

        new_token_start = token_position_map[old_token_start][0]
        new_token_end = token_position_map[old_token_end][0]

        entity['token_start'] = new_token_start
        entity['token_end'] = new_token_end

    if table_representation_method == 'concate':
        columns, column_info = get_columns_concate(example, table, bert_tokenizer)
    elif table_representation_method == 'canonical':
        columns, column_info = get_columns_canonical(example, table)
    else:
        raise RuntimeError('Unknown table representation')

    # gather table data
    for column in columns:
        column.name_tokens = bert_tokenizer.tokenize(str(column.name))
        column.sample_value_tokens = bert_tokenizer.tokenize(str(column.sample_value))

    rows = [table['kg'][row_id] for row_id in sorted(table['kg'])]
    valid_rows = []
    untokenized_rows = []
    for row in rows:
        valid_row = {}
        untokenized_row = {}
        for col in columns:
            cell_val = row.get(col.raw_name, [])
            if cell_val:
                cell_val = str(cell_val[0])
                untokenized_row[col.name] = cell_val
                cell_tokens = bert_tokenizer.tokenize(cell_val)
            else:
                cell_tokens = []
                untokenized_row[col.name] = ''

            valid_row[col.name] = cell_tokens

        valid_rows.append(valid_row)
        untokenized_rows.append(untokenized_row)

    table = Table(id=example['context'], header=columns, data=valid_rows, column_info=column_info)
    untokenized_table = Table(id=example['context'], header=columns, data=untokenized_rows)

    example['table'] = table
    example['untokenized_table'] = untokenized_table

    return example


def get_columns_canonical(example, table):
    # parse the table
    canonical_columns = OrderedDict()
    canonical_column_ids = OrderedDict()
    columns = []
    raw_column_canonical_ids = []
    for col_id, raw_column_name in enumerate(table['props']):
        column_name = raw_column_name[len('r.'):]
        type_pos = column_name.rfind('-')
        column_name = untyped_column_name = column_name[:type_pos]
        column_name = column_name.replace('-', ' ').replace('_', ' ')

        raw_type_string = raw_column_name[raw_column_name.rfind('-') + 1:]

        if raw_type_string == 'string':
            type_string = 'text'
        elif raw_type_string.startswith('num') or raw_type_string.startswith('date'):
            type_string = 'real'
        else:
            type_string = 'text'

        sample_value = get_sample_value(raw_column_name, table)

        if untyped_column_name in canonical_columns:
            column_entry = canonical_columns[untyped_column_name]

            if sample_value is not None and column_entry.type == 'text' and type_string == 'real':
                column_entry.type = 'real'
                column_entry.sample_value = sample_value

            raw_column_canonical_ids.append(canonical_column_ids[untyped_column_name])
        else:
            column = Column(name=column_name,
                            raw_name=raw_column_name,
                            type=type_string,
                            sample_value=sample_value)

            canonical_columns[untyped_column_name] = column
            canonical_column_ids[untyped_column_name] = col_id
            raw_column_canonical_ids.append(col_id)

        columns.append(
            Column(name=raw_column_name,
                   type=raw_type_string)
        )

    canonical_columns = list(canonical_columns.values())

    column_info = {
        'raw_columns': columns,
        'raw_column_canonical_ids': raw_column_canonical_ids
    }

    return canonical_columns, column_info


def get_columns_concate(example, table, bert_tokenizer):
    # parse the table
    columns = []
    for raw_column_name in table['props']:
        column_name = raw_column_name[len('r.'):]
        type_pos = column_name.rfind('-')
        column_name = column_name[:type_pos]
        column_name = column_name.replace('-', ' ').replace('_', ' ')

        type_string = raw_column_name[raw_column_name.rfind('-') + 1:]

        if type_string == 'string':
            type_string = 'text'
        elif type_string.startswith('num') or type_string.startswith('date'):
            type_string = 'real'
        else:
            type_string = 'text'

        sample_value, sample_value_tokens = get_sample_value(raw_column_name, table, bert_tokenizer)

        column = Column(name=raw_column_name,
                        type=type_string,
                        sample_value=sample_value,
                        name_tokens=bert_tokenizer.tokenize(column_name),
                        sample_value_tokens=sample_value_tokens)

        columns.append(column)

    return columns, {}


def get_sample_value(raw_column_name, table):
    sample_value = None
    for row_id, row in table['kg'].items():
        if raw_column_name in row and isinstance(row[raw_column_name], list) and len(str(row[raw_column_name][0])) > 0:
            sample_value = row[raw_column_name][0]
            break

    return sample_value


def load_environments(
    example_files: List[str],
    table_file: str,
    table_representation_method: str = 'canonical',
    example_ids: Iterable = None,
    bert_tokenizer: BertTokenizer = None
):
    dataset = []
    if example_ids is not None:
        example_ids = set(example_ids)

    for fn in example_files:
        data = load_jsonl(fn)
        for example in data:
            if example_ids:
                if example['id'] in example_ids:
                    dataset.append(example)
            else:
                dataset.append(example)

    print('{} examples in dataset.'.format(len(dataset)))

    tables = load_jsonl(table_file)
    table_dict = {table['name']: table for table in tables}
    print('{} tables.'.format(len(table_dict)))

    environments = create_environments(
        table_dict, dataset,
        table_representation_method=table_representation_method,
        executor_type='wtq',
        bert_tokenizer=bert_tokenizer,
    )
    print('{} environments in total'.format(len(environments)))

    return environments


def load_indexed_environments(
    file_patterns: Union[Iterable[Any], Path],
    table_file: Path,
    table_representation_method: Optional[str] = 'canonical',
) -> Dict[str, QAProgrammingEnv]:
    if isinstance(file_patterns, Path):
        file_patterns = [file_patterns]

    envs = load_environments(
        [str(f) for f in file_patterns],
        table_file=str(table_file),
        table_representation_method=table_representation_method,
        bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    )

    for env in envs:
        env.use_cache = False
        env.punish_extra_work = False

    env_dict = {
        env.name: env
        for env in envs
    }

    return env_dict


def create_environments(
    table_dict, dataset,
    table_representation_method,
    executor_type,
    max_n_mem=60, max_n_exp=3,
    bert_tokenizer=None
) -> List[QAProgrammingEnv]:
    all_envs = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print('creating environment #{}'.format(i))

        kg_info = table_dict[example['context']]

        env = create_environment(
            example, kg_info,
            table_representation_method,
            executor_type,
            max_n_mem, max_n_exp,
            bert_tokenizer
        )

        all_envs.append(env)

    return all_envs


def create_environment(
        example_dict: Dict, table_kg: Dict,
        table_representation_method: str,
        executor_type: str = 'wtq',
        max_n_mem: int = 60, max_n_exp: int = 3,
        bert_tokenizer: BertTokenizer = None,
) -> QAProgrammingEnv:
    if executor_type == 'wtq':
        score_fn = utils.wtq_score
        process_answer_fn = lambda x: x
        executor_fn = nsm.execution.worlds.wikitablequestions.WikiTableExecutor
    elif executor_type == 'wikisql':
        score_fn = utils.wikisql_score
        process_answer_fn = utils.wikisql_process_answer
        executor_fn = nsm.execution.worlds.wikisql.WikiSQLExecutor
    else:
        raise ValueError('Unknown executor {}'.format(executor_type))

    executor = executor_fn(table_kg)
    api = executor.get_api()
    type_hierarchy = api['type_hierarchy']
    func_dict = api['func_dict']
    constant_dict = api['constant_dict']

    interpreter = LispInterpreter(
        type_hierarchy=type_hierarchy,
        max_mem=max_n_mem,
        max_n_exp=max_n_exp,
        assisted=True
    )

    for v in func_dict.values():
        interpreter.add_function(**v)

    interpreter.add_constant(
        value=table_kg['row_ents'],
        type='entity_list',
        name='all_rows')

    if bert_tokenizer:
        example = annotate_example_for_bert(
            example_dict, table_kg, bert_tokenizer,
            table_representation_method=table_representation_method
        )

    env = QAProgrammingEnv(
        question_annotation=example,
        kg=table_kg,
        answer=process_answer_fn(example['answer']),
        constants=constant_dict.values(),
        interpreter=interpreter,
        score_fn=score_fn,
        name=example['id']
    )

    return env


def load_program_cache(cache_dir: Path) -> Dict:
    assert cache_dir.exists(), f'{str(cache_dir)} does not exsit!'

    program_cache: Dict = json.load(cache_dir.open())

    program_cache = {
        question_id: [
            hyp
            for hyp in hyp_list
            if hyp['prob'] is not None
        ]
        for question_id, hyp_list in program_cache.items()
        if (
            len([
                    hyp
                    for hyp in hyp_list
                    if hyp['prob'] is not None
            ]) > 0
        )
    }

    return program_cache


def to_human_readable_program(program, env):
    env = env.clone()
    env.use_cache = False
    ob = env.start_ob

    for tk in program:
        valid_actions = list(ob.valid_action_indices)
        action_id = env.de_vocab.lookup(tk)
        rel_action_id = valid_actions.index(action_id)
        ob, _, _, _ = env.step(rel_action_id)

    readable_program = []
    first_intermediate_var_id = len(
        [v for v, entry in env.interpreter.namespace.items() if v.startswith('v') and entry['is_constant']])
    for tk in program:
        if tk.startswith('v'):
            mem_entry = env.interpreter.namespace[tk]
            if mem_entry['is_constant']:
                if isinstance(mem_entry['value'], list):
                    token = mem_entry['value'][0]
                else:
                    token = mem_entry['value']
            else:
                intermediate_var_relative_id = int(tk[1:]) - first_intermediate_var_id
                token = 'v{}'.format(intermediate_var_relative_id)
        else:
            token = tk

        readable_program.append(token)

    return readable_program


def run_sample():
    envs = load_environments(["/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/data_split_1/train_split_shard_90-0.jsonl"],
                             "/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/tables.jsonl",
                             vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_vocab.json",
                             en_vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/en_vocab_min_count_5.json",
                             embedding_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_embedding_mat.npy")

    config = json.load(open('config.json'))
    agent = PGAgent.build(config)

    # agent.save(config['work_dir'] + '/model.bin')
    # agent2 = PGAgent.load(config['work_dir'] + '/model.bin')

    t1 = time.time()
    agent.beam_search(envs[:5], 32)
    t2 = time.time()
    print(t2 - t1)
    return

    for env in envs:
        agent.decode_examples([env], 10)

    t2 = time.time()
    print(t2 - t1)
    t2 = time.time()
    agent.beam_search(envs, 10)

    t3 = time.time()

    print(t3 - t2)


def inject_default_values(config: Dict):
    config.setdefault('table_representation', 'concate')
    config.setdefault('use_column_type_embedding', False)


def distributed_train(args):
    seed = int(args['--seed'])
    config_file = args['--config']
    use_cuda = args['--cuda']

    print(f'load config file [{config_file}]', file=sys.stderr)
    config = json.load(open(config_file))

    inject_default_values(config)

    if args['--extra-config'] != '{}':
        extra_config = args['--extra-config']
        print(f'load extra config [{extra_config}]', file=sys.stderr)
        extra_config = json.loads(extra_config)
        config.update(extra_config)

    work_dir = args['--work-dir']
    print(f'work dir [{work_dir}]', file=sys.stderr)
    config['work_dir'] = work_dir

    if not os.path.exists(work_dir):
        print(f'creating work dir [{work_dir}]', file=sys.stderr)
        os.makedirs(work_dir)

    json.dump(config, open(os.path.join(work_dir, 'config.json'), 'w'), indent=2)

    actor_use_table_bert_proxy = config.get('actor_use_table_bert_proxy', False)
    use_trainable_sketch_predictor = config.get('use_trainable_sketch_predictor', False)

    actor_devices = []
    evaluator_device = 'cpu'
    if use_cuda:
        print(f'use cuda', file=sys.stderr)
        device_count = torch.cuda.device_count()

        if use_trainable_sketch_predictor:
            assert device_count >= 3

            learner_devices = ['cuda:0', 'cuda:1']
            table_bert_server_device = 'cuda:2'
            sketch_predictor_device = 'cuda:2'
        else:
            assert device_count >= 2

            learner_devices = ['cuda:0', 'cuda:0']
            table_bert_server_device = 'cuda:1'
            sketch_predictor_device = 'cuda:1'

        evaluator_device = learner_devices[0]

        for i in range(2, device_count):
            actor_devices.append(f'cuda:{i}')
        else:
            actor_devices.append('cpu')
    else:
        learner_devices = [torch.device('cpu'), torch.device('cpu')]
        evaluator_device = torch.device('cpu')
        actor_devices.append(torch.device('cpu'))
        table_bert_server_device = torch.device('cpu')
        sketch_predictor_device = torch.device('cpu')

    shared_program_cache = SharedProgramCache()

    learner = Learner(
        config={**config, **{'seed': seed}},
        shared_program_cache=shared_program_cache,
        devices=learner_devices
    )

    print(f'Evaluator uses device {evaluator_device}', file=sys.stderr)
    evaluator = Evaluator(
        {**config, **{'seed': seed + 1}},
        eval_file=config['dev_file'], device=evaluator_device)
    learner.register_evaluator(evaluator)

    actor_num = config['actor_num']
    print('initializing %d actors' % actor_num, file=sys.stderr)
    actors = []
    # actor_shard_dict = {i: [] for i in range(actor_num)}
    train_shard_dir = Path(config['train_shard_dir'])
    shard_start_id = config['shard_start_id']
    shard_end_id = config['shard_end_id']
    train_example_ids = []
    for shard_id in range(shard_start_id, shard_end_id):
        shard_data = load_jsonl(train_shard_dir / f"{config['train_shard_prefix']}{shard_id}.jsonl")
        train_example_ids.extend(
            e['id']
            for e
            in shard_data
        )

        # actor_id = shard_id % actor_num
        # actor_shard_dict[actor_id].append(shard_id)

    per_actor_example_num = len(train_example_ids) // actor_num
    for actor_id in range(actor_num):
        actor = Actor(
            actor_id,
            example_ids=train_example_ids[
                actor_id * per_actor_example_num:
                ((actor_id + 1) * per_actor_example_num) if actor_id < actor_num - 1 else len(train_example_ids)
            ],
            shared_program_cache=shared_program_cache,
            device=actor_devices[actor_id % len(actor_devices)],
            config={**config, **{'seed': seed + 2 + actor_id}},)
        learner.register_actor(actor)

        actors.append(actor)

    if actor_use_table_bert_proxy:
        from nsm.parser_module.table_bert_proxy import TableBertServer

        table_bert_server = TableBertServer(config, table_bert_server_device)
        for actor in actors:
            table_bert_server.register_worker(actor)

        learner.register_table_bert_server(table_bert_server)

        print(f'starting table bert server @ {table_bert_server_device}', file=sys.stderr)
        table_bert_server.start()

    if use_trainable_sketch_predictor:
        from nsm.sketch.sketch_predictor import SketchPredictorServer

        sketch_predictor_server = SketchPredictorServer(config, sketch_predictor_device)
        for actor in actors:
            sketch_predictor_server.register_worker(actor)

        learner.register_sketch_predictor_server(sketch_predictor_server)
        print(f'starting sketch predictor server @ {sketch_predictor_device}', file=sys.stderr)
        sketch_predictor_server.start()

    # actors[0].run()
    print('starting %d actors' % actor_num, file=sys.stderr)
    for actor in actors:
        actor.start()
        pass

    print('starting evaluator', file=sys.stderr)
    evaluator.start()

    print('starting learner', file=sys.stderr)
    learner.start()

    # debug code
    # while True:
    #     for actor in actors:
    #         if not actor.is_alive():
    #             exit(0)
    #
    #     time.sleep(1)

    # while True:
    #     print('size of program cache', len(shared_program_cache.program_cache), file=sys.stderr)
    #     time.sleep(5)

    print('Learner process {}, evaluator process {}'.format(learner.pid, evaluator.pid), file=sys.stderr)

    # learner will quit first
    learner.join()
    print('Learner exited', file=sys.stderr)

    for actor in actors:
        actor.terminate()
        actor.join()

    if actor_use_table_bert_proxy:
        table_bert_server.terminate()
    if use_trainable_sketch_predictor:
        sketch_predictor_server.terminate()

    evaluator.terminate()
    evaluator.join()


def test(args):
    use_gpu = args['--cuda']
    model_path = args['--model']

    extra_config = json.loads(args['--extra-config'])
    if len(extra_config) > 0:
        print(f'load extra config [{extra_config}]', file=sys.stderr)

    print(f'loading model [{model_path}] for evaluation', file=sys.stderr)
    agent = PGAgent.load(model_path, gpu_id=0 if use_gpu else -1, **extra_config).eval()
    config = agent.config

    test_file = args['--test-file']
    print(f'loading test file [{test_file}]', file=sys.stderr)
    test_envs = load_environments(
        [test_file],
        table_file=config['table_file'],
        table_representation_method=config['table_representation'],
        bert_tokenizer=agent.encoder.bert_model.tokenizer
    )

    for env in test_envs:
        env.use_cache = False
        env.punish_extra_work = False

    # test_envs = load_environments([test_file],
    #                               table_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/tables.jsonl",
    #                               vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_vocab.json",
    #                               en_vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/en_vocab_min_count_5.json",
    #                               embedding_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_embedding_mat.npy")

    batch_size = int(args['--eval-batch-size'])
    beam_size = int(args['--eval-beam-size'])
    if beam_size == 0:
        beam_size = config['beam_size']
    print(f'batch size {batch_size}, beam size {beam_size}', file=sys.stderr)
    decode_results = agent.decode_examples(test_envs,
                                           beam_size=beam_size,
                                           batch_size=batch_size)
    assert len(test_envs) == len(decode_results)
    eval_results = Evaluation.evaluate_decode_results(test_envs, decode_results)
    print(eval_results, file=sys.stderr)

    save_to = args['--save-decode-to']
    if save_to != 'None':
        print(f'save results to [{save_to}]', file=sys.stderr)

        results = to_decode_results_dict(decode_results, test_envs)

        json.dump(results, open(save_to, 'w'), indent=2)


def to_decode_results_dict(decode_results, test_envs):
    results = OrderedDict()

    for env, hyp_list in zip(test_envs, decode_results):

        if hyp_list:
            table = hyp_list[0].logging_info['input_table']
            table = table.data
        else:
            table = None

        env_result = {
            'name': env.name,
            'question': ' '.join(str(x) for x in env.context['original_tokens']),
            'table': table,
            'hypotheses': None
        }

        hypotheses = []
        for hyp in hyp_list:
            hypotheses.append(OrderedDict(
                program=' '.join(str(x) for x in to_human_readable_program(hyp.trajectory.program, env)),
                # program=hyp.trajectory.program,
                is_correct=hyp.trajectory.reward == 1.,
                prob=hyp.prob
            ))

        env_result['hypotheses'] = hypotheses
        env_result['top_prediction_correct'] = hypotheses and hypotheses[0]['is_correct']
        results[env.name] = env_result

    return results


def main():
    multiprocessing.set_start_method('spawn', force=True)

    args = docopt(__doc__)

    if args['train']:
        distributed_train(args)
    elif args['test']:
        test(args)


def sanity_check():
    torch.manual_seed(123)
    np.random.seed(123 * 13 // 7)
    import random
    random.seed(123)

    envs = load_environments([
                                 "/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/data_split_1/train_split_shard_90-0.jsonl"],
                             "/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/tables.jsonl",
                             vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_vocab.json",
                             en_vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/en_vocab_min_count_5.json",
                             embedding_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_embedding_mat.npy")

    config = json.load(open('config.json'))
    agent = PGAgent.build(config).eval()

    t1 = time.time()
    for i in range(5):
        samples1 = agent.new_sample(envs[:5], sample_num=128)
    print('took %f s' % ((time.time() - t1) / 5))
    # print(samples1[0])

    t1 = time.time()
    for i in range(5):
        samples2 = agent.sample(envs[:5], sample_num=128)
    print('took %f s' % ((time.time() - t1) / 5))
    # print(samples2[0])

    #
    # buffer = AllGoodReplayBuffer(agent, envs[0].de_vocab)
    #
    # from nsm.actor import load_programs_to_buffer
    # load_programs_to_buffer(envs, buffer, config['saved_program_file'])
    #
    # trajs1 = buffer.trajectory_buffer[envs[0].name][:3]
    # trajs2 = buffer.trajectory_buffer[envs[1].name][:3]
    # trajs3 = buffer.trajectory_buffer[envs[2].name][:3]
    # trajs4 = buffer.trajectory_buffer[envs[3].name][:3]
    # agent.eval()
    #
    # batch_probs = agent(trajs1 + trajs2 + trajs3 + trajs4)
    #
    # print(batch_probs)
    # for traj in trajs1 + trajs2 + trajs3 + trajs4:
    #     single_prob = agent.compute_trajectory_prob([traj])
    #     print(single_prob)
    #
    # agent.decode_examples(envs[:1], beam_size=5)


def run_example():
    # envs = load_environments(["/Users/pengcheng/Research/datasets/wikitable/processed_input/train_examples.jsonl"],
    #                          "/Users/pengcheng/Research/datasets/wikitable/processed_input/tables.jsonl",
    #                          vocab_file="/Users/pengcheng/Research/datasets/wikitable/raw_input/wikitable_glove_vocab.json",
    #                          en_vocab_file="/Users/pengcheng/Research/datasets/wikitable/processed_input/preprocess_14/en_vocab_min_count_5.json",
    #                          embedding_file="/Users/pengcheng/Research/datasets/wikitable/raw_input/wikitable_glove_embedding_mat.npy")
    # #
    # env_dict = {env.name: env for env in envs}
    # env_dict['nt-3035'].interpreter.interactive(assisted=True)

    examples = load_jsonl(
        "/Users/pengcheng/Research/datasets/wikitable/processed_input/wtq_preprocess_revised/"
        "train_examples.jsonl")
    tables = load_jsonl(
        "/Users/pengcheng/Research/datasets/wikitable/processed_input/wtq_preprocess_revised/"
        "tables.jsonl")
    # # # #
    examples_dict = {e['id']: e for e in examples}
    tables_dict = {tab['name']: tab for tab in tables}
    # # # #
    q_id = 'nt-924' # 'nt-10767'
    interpreter = init_interpreter_for_example(examples_dict[q_id], tables_dict[examples_dict[q_id]['context']]).clone()
    interpreter.interactive(assisted=True)
    # program = ['(', 'argmax', 'all_rows', 'v4', ')', '(', 'hop', 'v8', 'v2', ')', '<END>']
    # for token in program:
    #     print(interpreter.valid_tokens())
    #     interpreter.read_token(token)
    # from table.wtq.evaluator import check_prediction
    # is_correct = utils.wtq_score([0.4], ['00.4', '0.4'])
    # print(is_correct)

if __name__ == '__main__':
    # envs = load_environments(["/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/train_examples.jsonl"],
    #                          "/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/tables.jsonl",
    #                          vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_vocab.json",
    #                          en_vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/processed_input/preprocess_14/en_vocab_min_count_5.json",
    #                          embedding_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_embedding_mat.npy")
    # #
    # env_dict = {env.name: env for env in envs}
    # env_dict['nt-3035'].interpreter.interactive(assisted=True)

    # examples = load_jsonl("/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/data_split_1/train_split.jsonl")
    # tables = load_jsonl("/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/wtq_preprocess/tables.jsonl")
    # # # # #
    # examples_dict = {e['id']: e for e in examples}
    # tables_dict = {tab['name']: tab for tab in tables}
    # # # # #
    # q_id = 'nt-3302'
    # interpreter = init_interpreter_for_example(examples_dict[q_id], tables_dict[examples_dict[q_id]['context']]).clone()
    # interpreter.interactive(assisted=True)
    # program = ['(', 'argmax', 'all_rows', 'v4', ')', '(', 'hop', 'v8', 'v2', ')', '<END>']
    # for token in program:
    #     print(interpreter.valid_tokens())
    #     interpreter.read_token(token)
    # # from table.wtq.evaluator import check_prediction
    # # is_correct = utils.wtq_score([0.4], ['00.4', '0.4'])
    # # print(is_correct)

    # run_sample()
    # run_example()
    main()
    # sanity_check()

