import heapq
import math
import os
import random
import sys
import time
import json
from typing import List

import numpy as np

import multiprocessing
from multiprocessing import Process

from nsm import nn_util
from nsm.agent_factory import PGAgent, Sample
from nsm.env_factory import Trajectory, Environment

import torch


def normalize_probs(p_list):
    smoothing = 1.e-8
    p_list = np.array(p_list) + smoothing

    return p_list / p_list.sum()


def load_programs_to_buffer(envs, replay_buffer, saved_programs_file_path):
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

    print('@' * 100)
    print('loading programs from file {}'.format(saved_programs_file_path))
    print('at least 1 solution found fraction: {}'.format(
        float(n_found) / total_env))

    replay_buffer.save_trajectories(trajectories)

    n_trajs_buffer = 0
    for k, v in replay_buffer._buffer.items():
        n_trajs_buffer += len(v)

    print('{} programs in the file'.format(n))
    print('{} programs extracted'.format(len(trajectories)))
    print('{} programs in the buffer'.format(n_trajs_buffer))
    print('@' * 100)


class AllGoodReplayBuffer(object):
    def __init__(self, agent=None, de_vocab=None, discount_factor=1.0, debug=False):
        self._buffer = dict()
        self.discount_factor = discount_factor
        self.agent = agent
        self.de_vocab = de_vocab
        self.program_prob_dict = dict()
        self.prob_sum_dict = dict()

    def has_found_solution(self, env_name):
        return env_name in self._buffer and self._buffer[env_name]

    def contains(self, traj: Trajectory):
        env_name = traj.environment_name
        if env_name not in self.program_prob_dict:
            return False

        program = traj.program
        if not program:
            program = self.de_vocab.lookup(traj.tgt_action_ids, reverse=True)

        program_str = ' '.join(program)

        if program_str in self.program_prob_dict[env_name]:
            return True
        else:
            return False

    @property
    def size(self):
        n = 0
        for _, v in self._buffer.items():
            n += len(v)
        return n

    def add_trajectory(self, trajectory: Trajectory, prob=None):
        program = trajectory.program
        if not program:
            program = self.de_vocab.lookup(trajectory.tgt_action_ids, reverse=True)

        program_str = ' '.join(program)

        self.program_prob_dict.setdefault(trajectory.environment_name, dict())[program_str] = prob

        self._buffer.setdefault(trajectory.environment_name, []).append(trajectory)

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
        for env, trajs in self._buffer.items():
            prob_dict = self.program_prob_dict[env]
            samples[env] = [Sample(traj, prob=prob_dict[' '.join(traj.program)]) for traj in trajs]

        return samples

    def replay(self, environments, n_samples=1, use_top_k=False, truncate_at_n=0, replace=True):
        select_env_names = set([e.name for e in environments])
        trajs = []

        # Collect all the trajs for the selected environments.
        for name in select_env_names:
            if name in self._buffer:
                trajs += self._buffer[name]

        if len(trajs) == 0:
            return []

        trajectory_probs = self.agent.compute_trajectory_prob(trajs, log=False)

        # Put the samples into an dictionary keyed by env names.
        samples = [Sample(trajectory=t, prob=p) for t, p in zip(trajs, trajectory_probs)]
        env_sample_dict = dict()
        for sample in samples:
            name = sample.trajectory.environment_name
            env_sample_dict.setdefault(name, []).append(sample)

        replay_samples = []
        for name, samples in env_sample_dict.items():
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
                self._buffer[name] = [sample.traj for sample in samples]

            # Compute the sum of prob of replays in the buffer.
            self.prob_sum_dict[name] = sum([sample.prob for sample in samples])

            for sample in samples:
                self.program_prob_dict[name][' '.join(sample.trajectory.program)] = sample.prob

            if use_top_k:
                # Select the top k samples weighted by their probs.
                selected_samples = heapq.nlargest(
                    n_samples, samples, key=lambda s: s.prob)
                # replay_samples += normalize_probs(selected_samples)
            else:
                # Randomly samples according to their probs.

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
    def __init__(self, actor_id, shard_ids, config):
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

    def run(self):
        def get_train_shard_path(i):
            return os.path.join(
                self.config['train_shard_dir'], self.config['train_shard_prefix'] + str(i) + '.jsonl')

        # load environments
        self.load_environments([get_train_shard_path(i) for i in self.shard_ids])

        # create agent and set it to evaluation mode
        self.agent = PGAgent.build(self.config).eval()
        nn_util.glorot_init(p for p in self.agent.parameters() if p.requires_grad)

        self.replay_buffer = AllGoodReplayBuffer(self.agent, self.environments[0].de_vocab)

        if self.config['load_saved_programs']:
            load_programs_to_buffer(self.environments, self.replay_buffer, self.config['saved_program_file'])
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
                                                               replace=config['replay_sample_with_replacement'])
                    t2 = time.time()
                    print(f'[Actor {self.actor_id}] epoch {epoch_id} batch {batch_id}, got {len(replay_samples)} replay samples (took {t2 - t1}s)',
                          file=sys.stderr)

                    samples_info = dict()
                    if method == 'mapo':
                        train_examples = []
                        for sample in replay_samples:
                            sample_weight = self.replay_buffer.prob_sum_dict.get(sample.trajectory.environment_name, 0.)
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
                            replay_samples_prob = self.replay_buffer.prob_sum_dict.get(sample.trajectory.environment_name, 0.)
                            if replay_samples_prob > 0.:
                                # clip the sum of probabilities for replay samples if the replay buffer is not empty
                                replay_samples_prob = max(replay_samples_prob, self.config['min_replay_samples_weight'])

                            sample_weight = 1. - replay_samples_prob

                            sample.weight = sample_weight * 1. / config['n_policy_samples']
                            train_examples.append(sample)

                        n_clip = 0
                        for env in batched_envs:
                            name = env.name
                            if (name in self.replay_buffer.prob_sum_dict and
                                    self.replay_buffer.prob_sum_dict[name] < self.config['min_replay_samples_weight']):
                                n_clip += 1
                        clip_frac = n_clip / len(batched_envs)

                        train_examples = train_examples
                        samples_info['clip_frac'] = clip_frac
                    elif method == 'mml':
                        for sample in replay_samples:
                            sample.weight = sample.prob / self.replay_buffer.prob_sum_dict[sample.trajectory.environment_name]
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

                buffer_content = dict()
                for env_name, samples in self.replay_buffer.all_samples().items():
                    buffer_content[env_name] = [dict(program=' '.join(sample.trajectory.program), prob=sample.prob) for sample in samples]
                buffer_save_path = os.path.join(config['work_dir'], f'replay_buffer_actor{self.actor_id}_epoch{epoch_id}.json')
                with open(buffer_save_path, 'w') as f:
                    json.dump(buffer_content, f, indent=2)

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