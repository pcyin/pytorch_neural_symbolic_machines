import json
import time
import os
import multiprocessing
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

import nsm.execution.worlds.wikitablequestions
from nsm import data_utils
from nsm import env_factory
from nsm.execution import executor_factory
from nsm import computer_factory
from table.utils import wtq_score


def get_experiment_dir():
    experiment_dir = args.output_dir / args.experiment_name

    return experiment_dir


def random_explore(env, use_cache=True, trigger_dict=None):
    env = env.clone()
    env.use_cache = use_cache
    question_tokens = env.question_annotation['tokens']
    if 'pos_tags' in env.question_annotation:
        pos_tags = env.question_annotation['pos_tags']
        tokens = question_tokens + pos_tags
    else:
        tokens = question_tokens
    invalid_functions = []
    if trigger_dict is not None:
        for function, trigger_words in trigger_dict.items():
            # if there is no overlap between `trigger_words` and `tokens`
            if not set(trigger_words) & set(tokens):
                invalid_functions.append(function)
    ob = env.start_ob
    while not env.done:
        invalid_actions = env.de_vocab.lookup(invalid_functions)
        valid_actions = ob.valid_action_indices
        new_valid_actions = list(set(valid_actions) - set(invalid_actions))
        # No action available anymore.
        if len(new_valid_actions) <= 0:
            return None
        new_action = np.random.randint(0, len(new_valid_actions))
        action = valid_actions.index(new_valid_actions[new_action])
        ob, _, _, _ = env.step(action)

    if sum(env.rewards) >= 1.0:
        return env.de_vocab.lookup(env.mapped_actions, reverse=True)
    else:
        return None


def run_random_exploration(shard_id):
    experiment_dir = get_experiment_dir()
    experiment_dir.mkdir(exist_ok=True, parents=True)

    if args.trigger_word_file.exists():
        with args.trigger_word_file.open() as f:
            trigger_dict = json.load(f)
            print('use trigger words in {}'.format(args.trigger_word_file))
    else:
        trigger_dict = None

    # Load dataset.
    train_set = []
    train_shard_file = Path(args.train_file_tmpl.format(shard_id))
    print('working on shard {}'.format(train_shard_file))
    with train_shard_file.open() as f:
        for line in f:
            example = json.loads(line)
            train_set.append(example)
    print('{} examples in training set.'.format(len(train_set)))

    table_dict = {}
    with args.table_file.open() as f:
        for line in f:
            table = json.loads(line)
            table_dict[table['name']] = table
    print('{} tables.'.format(len(table_dict)))

    if args.executor == 'wtq':
        score_fn = wtq_score
        process_answer_fn = lambda x: x
        executor_fn = nsm.execution.worlds.wikitablequestions.WikiTableExecutor
    elif args.executor == 'wikisql':
        raise NotImplementedError()
    else:
        raise ValueError('Unknown executor {}'.format(args.executor))

    all_envs = []
    t1 = time.time()
    for i, example in enumerate(train_set):
        if i % 100 == 0:
            print('creating environment #{}'.format(i))
        kg_info = table_dict[example['context']]
        executor = executor_fn(kg_info)
        api = executor.get_api()
        type_hierarchy = api['type_hierarchy']
        func_dict = api['func_dict']
        constant_dict = api['constant_dict']
        interpreter = computer_factory.LispInterpreter(
            type_hierarchy=type_hierarchy,
            max_mem=args.max_n_mem, max_n_exp=args.max_n_exp, assisted=True)
        for v in func_dict.values():
            interpreter.add_function(**v)

        interpreter.add_constant(
            value=kg_info['row_ents'], type='entity_list', name='all_rows')

        de_vocab = interpreter.get_vocab()
        env = env_factory.QAProgrammingEnv(
            data_utils.Vocab([]), de_vocab, question_annotation=example,
            answer=process_answer_fn(example['answer']),
            constants=constant_dict.values(),
            interpreter=interpreter,
            constant_value_embedding_fn=lambda x: None,
            score_fn=score_fn,
            max_cache_size=args.n_explore_samples * args.n_epoch * 10,
            name=example['id'])
        all_envs.append(env)

    program_dict = dict([(env.name, []) for env in all_envs])
    for i in range(1, args.n_epoch + 1):
        print('iteration {}'.format(i))
        t1 = time.time()
        for env in all_envs:
            for _ in range(args.n_explore_samples):
                program = random_explore(env, trigger_dict=trigger_dict)
                if program is not None:
                    program_dict[env.name].append(program)
        t2 = time.time()
        print('{} sec used in iteration {}'.format(t2 - t1, i))

        if i % args.save_every_n == 0 or i >= args.n_epoch:
            print('saving programs and cache in iteration {}'.format(i))
            t1 = time.time()
            with open(os.path.join(
                    get_experiment_dir(), 'program_shard_{}-{}.json'.format(shard_id, i)), 'w') as f:
                program_str_dict = dict([(k, [' '.join(p) for p in v]) for k, v
                                         in program_dict.items()])
                json.dump(program_str_dict, f, sort_keys=True, indent=2)

            # cache_dict = dict([(env.name, list(env.cache._set)) for env in all_envs])
            t2 = time.time()
            print(
                '{} sec used saving programs and cache in iteration {}'.format(
                    t2 - t1, i))

        n = len(all_envs)
        solution_ratio = len([env for env in all_envs if program_dict[env.name]]) * 1.0 / n
        print(
            'At least one solution found ratio: {}'.format(solution_ratio))
        n_programs_per_env = np.array([len(program_dict[env.name]) for env in all_envs])
        print(
            'number of solutions found per example: max {}, min {}, avg {}, std {}'.format(
                n_programs_per_env.max(), n_programs_per_env.min(), n_programs_per_env.mean(),
                n_programs_per_env.std()))

        # Macro average length.
        mean_length = np.mean([np.mean([len(p) for p in program_dict[env.name]]) for env in all_envs
                               if program_dict[env.name]])
        print('macro average program length: {}'.format(
            mean_length))
        # avg_cache_size = sum([len(env.cache._set) for env in all_envs]) * 1.0 / len(all_envs)
        # tf.logging.info('average cache size: {}'.format(
        #  avg_cache_size))


def collect_programs():
    saved_programs = {}
    for i in range(args.id_start, args.id_end):
        with open(os.path.join(
                get_experiment_dir(),
                'program_shard_{}-{}.json'.format(i, args.n_epoch)), 'r') as f:
            program_shard = json.load(f)
            saved_programs.update(program_shard)
    saved_program_path = get_experiment_dir() / 'saved_programs.json'
    with saved_program_path.open('w') as f:
        json.dump(saved_programs, f)
    print('saved programs are aggregated in {}'.format(saved_program_path))


def main(unused_argv):
    ps = []
    # run_random_exploration(0)
    for idx in range(args.id_start, args.id_end):
        p = multiprocessing.Process(target=run_random_exploration, args=(idx,))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
    collect_programs()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='output directory')
    parser.add_argument(
        '--experiment_name',
        type=str,
        required=True,
        help='All outputs of this experiment is'
             ' saved under a folder with the same name.')

    parser.add_argument('--table_file', type=Path, required=True, help='table file')
    parser.add_argument('--train_file_tmpl', type=str, required=True, help='training shards')
    parser.add_argument('--trigger_word_file', type=Path, required=False, help='trigger word file')

    parser.add_argument('--n_epoch', type=int, default=10)

    parser.add_argument('--max_n_mem', type=int, default=100, help='Max number of memory slots in the "computer".')
    parser.add_argument('--max_n_exp', type=int, default=3, help='Max number of expressions allowed in a program.')
    parser.add_argument('--max_n_valid_indices', type=int, default=100,
                        help='Max number of valid tokens during decoding.')
    parser.add_argument('--executor', type=str, default='wtq', help='Which executor to use, wtq or wikisql.')

    parser.add_argument('--n_explore_samples', type=int, default=50)
    parser.add_argument('--save_every_n', type=int, default=10)
    parser.add_argument('--id_start', type=int, default=0)
    parser.add_argument('--id_end', type=int, default=0)

    args = parser.parse_args()

    main(args)
