import heapq
import math
import os
import random
import sys
import time
import json
from typing import List, Dict

import numpy as np

import multiprocessing
from multiprocessing import Process

from nsm import nn_util
from nsm.agent_factory import PGAgent, Sample
from nsm.consistency_utils import ConsistencyModel, QuestionSimilarityModel
from nsm.env_factory import Trajectory, Environment

import torch

from nsm.program_cache import SharedProgramCache
from nsm.dist_util import STOP_SIGNAL


def normalize_probs(p_list):
    smoothing = 1.e-8
    p_list = np.array(p_list) + smoothing

    return p_list / p_list.sum()


class ReplayBuffer(object):
    def __init__(self, agent, shared_program_cache, discount_factor=1.0, debug=False):
        self.trajectory_buffer = dict()
        self.discount_factor = discount_factor
        self.agent = agent
        self.shared_program_cache = shared_program_cache
        self.env_program_prob_dict = dict()
        self.env_program_prob_sum_dict = dict()

    def load(self, envs: List[Environment], saved_programs_file_path: str):
        programs = json.load(open(saved_programs_file_path))

        trajectories = []
        n = 0
        total_env = 0
        n_found = 0
        for env in envs:
            total_env += 1
            found = False
            if env.name in programs:
                program_str_list = programs[env.name]
                n += len(program_str_list)
                env.cache._set = set(program_str_list)
                for program_str in program_str_list:
                    program = program_str.split()
                    try:
                        traj = Trajectory.from_program(env, program)
                    except ValueError:
                        print(f'Error loading program {program} for env {env.name}', file=sys.stderr)
                        continue

                    if traj is not None:
                        trajectories.append(traj)
                        found = True
                        n_found += 1

        print('@' * 100, file=sys.stderr)
        print('loading programs from file {}'.format(saved_programs_file_path), file=sys.stderr)
        print('at least 1 solution found fraction: {}'.format(
            float(n_found) / total_env), file=sys.stderr)

        self.save_trajectories(trajectories)
        print('{} programs in the file'.format(n), file=sys.stderr)
        print('{} programs extracted'.format(len(trajectories)), file=sys.stderr)
        print('{} programs in the buffer'.format(self.program_num), file=sys.stderr)
        print('@' * 100, file=sys.stderr)

    def has_found_solution(self, env_name):
        return env_name in self.trajectory_buffer and self.trajectory_buffer[env_name]

    def contains(self, traj: Trajectory):
        env_name = traj.environment_name
        if env_name not in self.trajectory_buffer:
            return False

        program = traj.program
        program_str = ' '.join(program)

        if program_str in self.env_program_prob_dict[env_name]:
            return True
        else:
            return False

    @property
    def size(self):
        n = 0
        for _, v in self.trajectory_buffer.items():
            n += len(v)
        return n

    @property
    def program_num(self):
        return sum(len(v) for v in self.env_program_prob_dict.values())

    def update_program_prob(self, env_name, program: List[str], prob: float):
        self.env_program_prob_dict[env_name][' '.join(program)] = prob
        self.shared_program_cache.update_hypothesis(env_name, program, prob)

    def add_trajectory(self, trajectory: Trajectory, prob=None):
        program = trajectory.program

        self.shared_program_cache.add_hypothesis(trajectory.environment_name, program, prob)
        self.env_program_prob_dict.setdefault(trajectory.environment_name, dict())[' '.join(program)] = prob

        self.trajectory_buffer.setdefault(trajectory.environment_name, []).append(trajectory)

    def save_trajectories(self, trajectories):
        for trajectory in trajectories:
            # program_str = ' '.join(trajectory.to_program)
            # if trajectory.reward == 1.:
            if not self.contains(trajectory):
                # print(f'add 1 traj [{trajectory}] for env [{trajectory.environment_name}] to buffer', file=sys.stderr)
                self.add_trajectory(trajectory)

    def save_samples(self, samples: List[Sample], log=True):
        for sample in samples:
            if not self.contains(sample.trajectory):
                prob = math.exp(sample.prob) if log else sample.prob
                self.add_trajectory(sample.trajectory, prob=prob)

    def all_samples(self, agent=None):
        samples = dict()
        for env_name, trajs in self.trajectory_buffer.items():
            samples[env_name] = [Sample(traj, prob=self.env_program_prob_dict[env_name][' '.join(traj.program)]) for traj in trajs]

        return samples

    def replay(self, environments, n_samples=1, use_top_k=False, truncate_at_n=0, replace=True, consistency_model: ConsistencyModel = None):
        select_env_names = set([e.name for e in environments])
        trajs = []

        # Collect all the trajs for the selected environments.
        for env_name in select_env_names:
            if env_name in self.trajectory_buffer:
                trajs += self.trajectory_buffer[env_name]

        if len(trajs) == 0:
            return []

        trajectory_probs = self.agent.compute_trajectory_prob(trajs, log=False)

        # Put the samples into an dictionary keyed by env names.
        samples = [Sample(trajectory=t, prob=p) for t, p in zip(trajs, trajectory_probs)]
        env_sample_dict = dict()
        for sample in samples:
            env_name = sample.trajectory.environment_name
            env_sample_dict.setdefault(env_name, []).append(sample)

        replay_samples = []
        for env_name, samples in env_sample_dict.items():
            n = len(samples)

            # Truncated the number of samples in the selected
            # samples and in the buffer.
            if 0 < truncate_at_n < n:
                # Randomize the samples before truncation in case
                # when no prob information is provided and the trajs
                # need to be truncated randomly.
                random.shuffle(samples)
                samples = heapq.nlargest(
                    truncate_at_n, samples, key=lambda s: s.prob)
                self.trajectory_buffer[env_name] = [sample.traj for sample in samples]

            # Compute the sum of prob of replays in the buffer.
            self.env_program_prob_sum_dict[env_name] = sum([sample.prob for sample in samples])

            for sample in samples:
                self.update_program_prob(env_name, sample.trajectory.program, sample.prob)

            if use_top_k:
                # Select the top k samples weighted by their probs.
                selected_samples = heapq.nlargest(
                    n_samples, samples, key=lambda s: s.prob)
                # replay_samples += normalize_probs(selected_samples)
            else:
                # Randomly samples according to their probs.

                if consistency_model:
                    # log_p_samples = np.log([sample.prob for sample in samples])
                    p_samples = consistency_model.compute_consistency_and_rescore(env_name, samples)
                    # p_samples = consistency_model.rescore(log_p_samples, consistency_scores)
                else:
                    p_samples = normalize_probs([sample.prob for sample in samples])

                if replace:
                    selected_sample_indices = np.random.choice(len(samples), n_samples, p=p_samples)
                else:
                    sample_num = min(len(samples), n_samples)
                    selected_sample_indices = np.random.choice(len(samples), sample_num, p=p_samples, replace=False)

                selected_samples = [samples[i] for i in selected_sample_indices]

            selected_samples = [Sample(trajectory=sample.trajectory, prob=sample.prob) for sample in selected_samples]
            replay_samples += selected_samples

        return replay_samples


class Actor(Process):
    def __init__(self, actor_id, shard_ids, shared_program_cache, config):
        super(Actor, self).__init__(daemon=True)

        # self.checkpoint_queue = checkpoint_queue
        # self.eval_queue = eval_queue
        # self.replay_queue = replay_queue

        self.config = config
        self.actor_id = actor_id
        self.shard_ids = shard_ids

        if not self.shard_ids:
            raise RuntimeError(f'empty shard for Actor {self.actor_id}')

        self.model_path = None
        self.checkpoint_queue = None
        self.train_queue = None
        self.shared_program_cache = shared_program_cache
        self.consistency_model = None

    @property
    def use_consistency_model(self):
        return self.config['use_consistency_model']

    def run(self):
        def get_train_shard_path(i):
            return os.path.join(
                self.config['train_shard_dir'], self.config['train_shard_prefix'] + str(i) + '.jsonl')

        # load environments
        self.load_environments([get_train_shard_path(i) for i in self.shard_ids])

        if self.use_consistency_model:
            print('Load consistency model', file=sys.stderr)
            self.consistency_model = ConsistencyModel(QuestionSimilarityModel.load(self.config['question_similarity_model_path']),
                                                      self.shared_program_cache,
                                                      self.environments,
                                                      alpha=float(self.config['consistency_alpha']),
                                                      log_file=os.path.join(self.config['work_dir'], f'consistency_model_actor_{self.actor_id}.log'),
                                                      debug=self.actor_id == 0)

        # create agent and set it to evaluation mode
        self.agent = PGAgent.build(self.config).eval()
        nn_util.glorot_init(p for p in self.agent.parameters() if p.requires_grad)

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

        with torch.no_grad():
            while True:
                epoch_id += 1
                epoch_start = time.time()
                batch_iter = nn_util.batch_iter(self.environments, batch_size=self.config['batch_size'], shuffle=True)
                for batch_id, batched_envs in enumerate(batch_iter):
                    # print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}', file=sys.stderr)
                    # perform sampling
                    t1 = time.time()
                    if sample_method == 'sample':
                        explore_samples = self.agent.sample(batched_envs,
                                                            sample_num=config['n_explore_samples'],
                                                            use_cache=config['use_cache'])
                    else:
                        explore_samples = self.agent.new_beam_search(batched_envs,
                                                                     beam_size=config['n_explore_samples'],
                                                                     use_cache=config['use_cache'],
                                                                     return_list=True)
                    t2 = time.time()
                    print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}, sampled {len(explore_samples)} trajectories (took {t2 - t1}s)', file=sys.stderr)

                    # retain samples with high reward
                    good_explore_samples = [sample for sample in explore_samples if sample.trajectory.reward == 1.]
                    # for sample in good_explore_samples:
                    #     print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}, '
                    #           f'add 1 traj [{sample.trajectory}] for env [{sample.trajectory.environment_name}] to buffer',
                    #           file=sys.stderr)
                    self.replay_buffer.save_samples(good_explore_samples)

                    # sample replay examples from the replay buffer
                    t1 = time.time()
                    replay_samples = self.replay_buffer.replay(batched_envs,
                                                               n_samples=config['n_replay_samples'],
                                                               use_top_k=config['use_top_k_replay_samples'],
                                                               replace=config['replay_sample_with_replacement'],
                                                               consistency_model=self.consistency_model)
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
                    else:
                        train_examples = replay_samples
                        for sample in train_examples:
                            sample.weight = 1.

                    if train_examples:
                        self.train_queue.put((train_examples, samples_info))
                    else:
                        continue

                    self.check_and_load_new_model()

                epoch_end = time.time()
                print(f"[Actor {self.actor_id}] epoch {epoch_id} finished, took {epoch_end - epoch_start}s", file=sys.stderr)

                # buffer_content = dict()
                # for env_name, samples in self.replay_buffer.all_samples().items():
                #     buffer_content[env_name] = [dict(program=' '.join(sample.trajectory.program), prob=sample.prob) for sample in samples]
                # buffer_save_path = os.path.join(config['work_dir'], f'replay_buffer_actor{self.actor_id}_epoch{epoch_id}.json')
                # with open(buffer_save_path, 'w') as f:
                #     json.dump(buffer_content, f, indent=2)
                if self.consistency_model:
                    self.consistency_model.log_file.flush()

    def load_environments(self, file_paths):
        from table.experiments import load_environments, create_environments
        envs = load_environments(file_paths,
                                 table_file=self.config['table_file'],
                                 vocab_file=self.config['vocab_file'],
                                 en_vocab_file=self.config['en_vocab_file'],
                                 embedding_file=self.config['embedding_file'])

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
            self.agent.load_state_dict(state_dict)
            self.model_path = new_model_path

            t2 = time.time()
            print('[Actor %d] loaded new model [%s] (took %.2f s)' % (self.actor_id, new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False