import os
import re
import sys
import time
import json
from pathlib import Path

import torch.multiprocessing as torch_mp
import multiprocessing

from nsm import nn_util
from nsm.parser_module import get_parser_agent_by_name
from nsm.parser_module.agent import PGAgent
from nsm.parser_module.sketch_guided_agent import SketchGuidedAgent
from nsm.consistency_utils import ConsistencyModel, QuestionSimilarityModel

import torch

from nsm.replay_buffer import ReplayBuffer
from nsm.sketch.sketch import SketchManager
from nsm.sketch.sketch_predictor import SketchPredictor, SketchPredictorProxy


class Actor(torch_mp.Process):
    def __init__(self, actor_id, example_ids, shared_program_cache, device, config):
        super(Actor, self).__init__(daemon=True)

        # self.checkpoint_queue = checkpoint_queue
        # self.eval_queue = eval_queue
        # self.replay_queue = replay_queue

        self.config = config
        self.actor_id = f'Actor_{actor_id}'
        self.example_ids = example_ids
        self.device = device

        if not self.example_ids:
            raise RuntimeError(f'empty shard for Actor {self.actor_id}')

        self.model_path = None
        self.checkpoint_queue = None
        self.train_queue = None
        self.shared_program_cache = shared_program_cache
        self.consistency_model = None

        if config.get('actor_use_table_bert_proxy', False):
            self.table_bert_result_queue = multiprocessing.Queue()

    @property
    def use_consistency_model(self):
        return self.config['use_consistency_model']

    @property
    def use_sketch_exploration(self):
        return self.config.get('use_sketch_exploration', False)

    @property
    def use_sketch_guided_replay(self):
        return self.config.get('use_sketch_guided_replay', False)

    def run(self):
        # initialize cuda context
        self.device = torch.device(self.device)
        if 'cuda' in self.device.type:
            torch.cuda.set_device(self.device)

        # seed the random number generators
        nn_util.init_random_seed(self.config['seed'], self.device)

        def get_train_shard_path(i):
            return os.path.join(
                self.config['train_shard_dir'], self.config['train_shard_prefix'] + str(i) + '.jsonl')

        # create agent and set it to evaluation mode
        if 'cuda' in str(self.device.type):
            torch.cuda.set_device(self.device)

        agent_name = self.config.get('parser', 'vanilla')
        self.agent = get_parser_agent_by_name(agent_name).build(self.config, master=self.actor_id).to(self.device).eval()

        # initialize sketch predictor
        use_trainable_sketch_predictor = self.config.get('use_trainable_sketch_predictor', False)
        if use_trainable_sketch_predictor:
            self.sketch_predictor = SketchPredictorProxy()
            self.sketch_predictor.initialize(self)

        if self.config.get('actor_use_table_bert_proxy', False):
            # initialize proxy
            self.agent.encoder.bert_model.initialize(self)

        # load environments
        self.load_environments(
            [
                get_train_shard_path(i)
                for i
                in range(self.config['shard_start_id'], self.config['shard_end_id'])
            ],
            example_ids=self.example_ids
        )

        if self.use_consistency_model:
            print('Load consistency model', file=sys.stderr)
            self.consistency_model = ConsistencyModel(QuestionSimilarityModel.load(self.config['question_similarity_model_path']),
                                                      self.shared_program_cache,
                                                      self.environments,
                                                      alpha=float(self.config['consistency_alpha']),
                                                      log_file=os.path.join(self.config['work_dir'], f'consistency_model_actor_{self.actor_id}.log'),
                                                      debug=self.actor_id == 0)

        self.replay_buffer = ReplayBuffer(self.agent, self.shared_program_cache)

        if self.config['load_saved_programs']:
            self.replay_buffer.load(self.environments, self.config['saved_program_file'])
            print(f'[Actor {self.actor_id}] loaded {self.replay_buffer.size} programs to buffer', file=sys.stderr)

        self.train()

    def train(self):
        config = self.config
        epoch_id = 0
        env_dict = {env.name: env for env in self.environments}
        sample_method = self.config['sample_method']
        method = self.config['method']
        assert sample_method in ('sample', 'beam_search')
        assert method in ('sample', 'mapo', 'mml')

        work_dir = Path(self.config['work_dir'])
        log_dir = work_dir / 'log'
        log_dir.mkdir(exist_ok=True, parents=True)

        debug_file = None
        if self.config.get('save_actor_log', False):
            debug_file = (log_dir / f'debug.actor{self.actor_id}.log').open('w')
        # self.agent.log = debug_file

        with torch.no_grad():
            while True:
                epoch_id += 1
                epoch_start = time.time()
                batch_iter = nn_util.batch_iter(self.environments, batch_size=self.config['batch_size'], shuffle=True)
                for batch_id, batched_envs in enumerate(batch_iter):
                    try:
                        # print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}', file=sys.stderr)
                        # perform sampling

                        strict_constraint_on_sketches = config.get('sketch_explore_strict_constraint_on_sketch', True)
                        force_sketch_coverage = config.get('sketch_explore_force_coverage', False)
                        constraint_sketches = None

                        if isinstance(self.agent, PGAgent) and self.use_sketch_exploration:
                            constraint_sketches = dict()
                            explore_beam_size = config.get('sketch_explore_beam_size', 5)
                            num_sketches_per_example = config.get('num_candidate_sketches', 5)
                            remove_explored_sketch = config.get('remove_explored_sketch', True)
                            use_sketch_exploration_for_nepoch = config.get('use_sketch_exploration_for_nepoch', 10000)
                            use_trainable_sketch_predictor = self.config.get('use_trainable_sketch_predictor', False)

                            if epoch_id <= use_sketch_exploration_for_nepoch:
                                t1 = time.time()
                                if use_trainable_sketch_predictor:
                                    candidate_sketches = self.sketch_predictor.get_sketches(
                                        batched_envs,
                                        K=num_sketches_per_example
                                    )
                                    for env, sketches in zip(batched_envs, candidate_sketches):
                                        constraint_sketches[env.name] = sketches
                                else:
                                    for env in batched_envs:
                                        env_candidate_sketches = self.sketch_predictor.get_sketches_from_similar_questions(
                                            env.name,
                                            remove_explored=remove_explored_sketch,
                                            log_file=None
                                        )

                                        if debug_file:
                                            print(f"Question {env.name} Candidate sketches in the cache:\n"
                                                  f"{json.dumps({str(k): v for k, v in env_candidate_sketches.items()}, indent=2, default=str)}", file=debug_file)

                                        env_candidate_sketches = sorted(
                                            env_candidate_sketches,
                                            key=lambda s: env_candidate_sketches[s]['score'],
                                            reverse=True)[:num_sketches_per_example]

                                    constraint_sketches[env.name] = env_candidate_sketches

                                # logging
                                # print('[Actor] Sampled sketches', file=sys.stderr)
                                # print(constraint_sketches, file=sys.stderr)
                                if debug_file:
                                    print(f'Found candidate sketches took {time.time() - t1}s', file=debug_file)
                                    for env in batched_envs:
                                        print("======", file=debug_file)
                                        print(f"Question [{env.name}] "
                                              f"{env.question_annotation['question']}", file=debug_file)

                                        print(
                                            f"Selected sketches for [{env.name}]:\n"
                                            f"{json.dumps(constraint_sketches[env.name], indent=2, default=str)}",
                                            file=debug_file
                                        )

                        t1 = time.time()
                        if sample_method == 'sample':
                            explore_samples = self.agent.sample(
                                batched_envs,
                                sample_num=config['n_explore_samples'],
                                use_cache=config['use_cache'],
                                constraint_sketches=constraint_sketches
                            )
                        else:
                            explore_samples = self.agent.new_beam_search(
                                batched_envs,
                                beam_size=config['n_explore_samples'],
                                use_cache=config['use_cache'],
                                return_list=True,
                                constraint_sketches=constraint_sketches,
                                strict_constraint_on_sketches=strict_constraint_on_sketches,
                                force_sketch_coverage=force_sketch_coverage
                            )
                        t2 = time.time()

                        if debug_file:
                            print('Explored programs:', file=debug_file)
                            for sample in explore_samples:
                                print(f"[{sample.trajectory.environment_name}] "
                                      f"{' '.join(sample.trajectory.program)} "
                                      f"(prob={sample.prob:.4f}, correct={sample.trajectory.reward == 1.})",
                                      file=debug_file)

                        print(
                            f'[Actor {self.actor_id}] '
                            f'epoch {epoch_id} batch {batch_id}, '
                            f'sampled {len(explore_samples)} trajectories (took {t2 - t1}s)', file=sys.stderr
                        )

                        # retain samples with high reward
                        good_explore_samples = [sample for sample in explore_samples if sample.trajectory.reward == 1.]
                        # for sample in good_explore_samples:
                        #     print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}, '
                        #           f'add 1 traj [{sample.trajectory}] for env [{sample.trajectory.environment_name}] to buffer',
                        #           file=sys.stderr)
                        self.replay_buffer.save_samples(good_explore_samples)

                        # sample replay examples from the replay buffer
                        t1 = time.time()
                        replay_constraint_sketches = None
                        if self.use_sketch_guided_replay:
                            replay_constraint_sketches = dict()
                            num_sketches_per_example = config.get('num_candidate_sketches', 5)

                            env_candidate_sketches = self.sketch_predictor.get_sketches(batched_envs)
                            env_selected_candidate_sketches = sorted(
                                env_candidate_sketches,
                                key=lambda s: env_candidate_sketches[s]['score'],
                                reverse=True)[:num_sketches_per_example]

                            replay_constraint_sketches[env.name] = env_selected_candidate_sketches

                            if debug_file:
                                for env in batched_envs:
                                    print("======begin sketch guided reply======", file=debug_file)
                                    print(f"Question [{env.name}] "
                                          f"{env.question_annotation['question']}", file=debug_file)

                                    print(
                                        f"Candidate sketches in the cache:\n"
                                        f"{json.dumps({str(k): v for k, v in env_candidate_sketches.items()}, indent=2, default=str)}",
                                        file=debug_file
                                    )

                                    print("======end sketch guided reply======", file=debug_file)

                        replay_samples = self.replay_buffer.replay(
                            batched_envs,
                            n_samples=config['n_replay_samples'],
                            use_top_k=config['use_top_k_replay_samples'],
                            replace=config['replay_sample_with_replacement'],
                            truncate_at_n=config.get('sample_replay_from_topk', 0),
                            consistency_model=self.consistency_model,
                            constraint_sketches=replay_constraint_sketches,
                            debug_file=debug_file
                        )
                        t2 = time.time()
                        print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}, got {len(replay_samples)} replay samples (took {t2 - t1}s)',
                              file=sys.stderr)

                        samples_info = dict()
                        if method == 'mapo':
                            train_examples = []
                            for sample in replay_samples:
                                sample_weight = self.replay_buffer.env_program_prob_sum_dict.get(sample.trajectory.environment_name, 0.)
                                sample_weight = max(sample_weight, self.config['min_replay_samples_weight'])

                                sample.weight = sample_weight * 1. / config['n_replay_samples']
                                train_examples.append(sample)

                            on_policy_samples = self.agent.sample(batched_envs,
                                                                  sample_num=config['n_policy_samples'],
                                                                  use_cache=False)
                            non_replay_samples = [sample for sample in on_policy_samples
                                                  if sample.trajectory.reward == 1. and not self.replay_buffer.contains(sample.trajectory)]
                            self.replay_buffer.save_samples(non_replay_samples)

                            for sample in non_replay_samples:
                                if self.use_consistency_model and self.consistency_model.debug:
                                    print(f'>>>>>>>>>> non replay samples for {sample.trajectory.environment_name}', file=self.consistency_model.log_file)
                                    self.consistency_model.compute_consistency_score(sample.trajectory.environment_name, [sample])
                                    print(f'<<<<<<<<<<< non replay samples for {sample.trajectory.environment_name}',
                                          file=self.consistency_model.log_file)

                                replay_samples_prob = self.replay_buffer.env_program_prob_sum_dict.get(sample.trajectory.environment_name, 0.)
                                if replay_samples_prob > 0.:
                                    # clip the sum of probabilities for replay samples if the replay buffer is not empty
                                    replay_samples_prob = max(replay_samples_prob, self.config['min_replay_samples_weight'])

                                sample_weight = 1. - replay_samples_prob

                                sample.weight = sample_weight * 1. / config['n_policy_samples']
                                train_examples.append(sample)

                            n_clip = 0
                            for env in batched_envs:
                                name = env.name
                                if (name in self.replay_buffer.env_program_prob_dict and
                                        self.replay_buffer.env_program_prob_sum_dict.get(name, 0.) < self.config['min_replay_samples_weight']):
                                    n_clip += 1
                            clip_frac = n_clip / len(batched_envs)

                            train_examples = train_examples
                            samples_info['clip_frac'] = clip_frac
                        elif method == 'mml':
                            for sample in replay_samples:
                                sample.weight = sample.prob / self.replay_buffer.env_program_prob_sum_dict[sample.trajectory.environment_name]
                            train_examples = replay_samples
                        elif method == 'sample':
                            train_examples = replay_samples
                            for sample in train_examples:
                                sample.weight = max(sample.prob, config['min_replay_samples_weight'])
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            msg = (
                                    f'[Actor {self.actor_id}] WARNING: ran out of memory with exception: '
                                    + '{};'.format(e)
                                    + '\n Skipping batch'
                            )
                            print(msg, file=sys.stderr)
                            sys.stderr.flush()

                            continue
                        else:
                            raise e

                    if train_examples:
                        self.train_queue.put((train_examples, samples_info))
                    else:
                        continue

                    self.check_and_load_new_model()
                    if debug_file:
                        debug_file.flush()

                    if self.device.type == 'cuda':
                        mem_cached_mb = torch.cuda.memory_cached() / 1000000
                        if mem_cached_mb > 8000:
                            print(f'Actor {self.actor_id} empty cached memory [{mem_cached_mb} MB]', file=sys.stderr)
                            torch.cuda.empty_cache()

                epoch_end = time.time()
                print(f"[Actor {self.actor_id}] epoch {epoch_id} finished, took {epoch_end - epoch_start}s", file=sys.stderr)

                # buffer_content = dict()
                # for env_name, samples in self.replay_buffer.all_samples().items():
                #     buffer_content[env_name] = [dict(program=' '.join(sample.trajectory.program), prob=sample.prob) for sample in samples]
                # buffer_save_path = os.path.join(config['work_dir'], f'replay_buffer_actor{self.actor_id}_epoch{epoch_id}.json')
                # with open(buffer_save_path, 'w') as f:
                #     json.dump(buffer_content, f, indent=2)

                # dump program cache for the current actor
                # cur_program_cache = self.replay_buffer.all_samples()
                # with multiprocessing.Lock():
                #     program_cache_save_file = log_dir / f'program_cache.epoch{epoch_id}.jsonl'
                #
                #     with program_cache_save_file.open('a') as f:
                #         for env_name, samples in cur_program_cache.items():
                #             entry = {
                #                 'question_id': env_name,
                #                 'hypotheses': [
                #                     {
                #                         'program': ' '.join(sample.trajectory.human_readable_program),
                #                         'prob': sample.prob
                #                     }
                #                     for sample in samples
                #                 ]
                #             }
                #             line = json.dumps(entry)
                #             f.write(line + os.linesep)

                if self.consistency_model:
                    self.consistency_model.log_file.flush()
                    sys.stderr.flush()

    def load_environments(self, file_paths, example_ids=None):
        from table.experiments import load_environments
        envs = load_environments(file_paths,
                                 example_ids=example_ids,
                                 table_file=self.config['table_file'],
                                 table_representation_method=self.config['table_representation'],
                                 bert_tokenizer=self.agent.encoder.bert_model.tokenizer)

        setattr(self, 'environments', envs)

    def check_and_load_new_model(self):
        t1 = time.time()
        while True:
            new_model_path = self.checkpoint_queue.get()
            # if new_model_path == STOP_SIGNAL:
            #     print(f'[Actor {self.actor_id}] Exited', file=sys.stderr)
            #     sys.stdout.flush()
            #     sys.stderr.flush()
            #     exit(0)

            if new_model_path == self.model_path or os.path.exists(new_model_path):
                break
        print(f'[Actor {self.actor_id}] {time.time() - t1}s used to wait for new checkpoint', file=sys.stderr)

        if new_model_path != self.model_path:
            t1 = time.time()

            state_dict = torch.load(new_model_path, map_location=lambda storage, loc: storage)
            self.agent.load_state_dict(state_dict, strict=False)
            self.model_path = new_model_path

            t2 = time.time()
            print('[Actor %s] loaded new model [%s] (took %.2f s)' % (self.actor_id, new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False

    def get_global_step(self):
        if not self.model_path:
            return 0

        model_name = self.model_path.split('/')[-1]
        train_iter = re.search('iter(\d+)?', model_name).group(1)

        return int(train_iter)
