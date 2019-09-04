import ctypes
import heapq
import os
import random
import time
from collections import OrderedDict
from itertools import chain
from typing import List, Dict

import numpy as np
import sys

import multiprocessing
from multiprocessing import Process

from nsm import nn_util
from nsm.parser_module.agent import PGAgent
from nsm.env_factory import Trajectory, QAProgrammingEnv, Sample

import torch
from tensorboardX import SummaryWriter

# retrain the model
from typing import List, Dict
import numpy as np
from nsm import nn_util

import editdistance


def load_nearest_neighbors(file_path):
    nns = dict()
    with open(file_path) as f:
        for line in f:
            q_ids = line.strip().split('\t')
            query_qid = q_ids[0]
            neighbors = q_ids[1].split(',')
            nns[query_qid] = neighbors

    return nns


def get_canonical_program_signature(program):
    stack = []
    i = 0
    while i < len(program):
        token = program[i]
        if token == '(':
            new_expr = []
            stack.append(new_expr)
        elif token == ')':
            pass
        elif token == '<END>':
            pass
        else:
            stack[-1].append(token)

        i += 1

    program_sig = [expr[0] for expr in stack]

    return program_sig


def compute_program_similarity(program1, program2):
    program_sig1 = get_canonical_program_signature(program1)
    program_sig2 = get_canonical_program_signature(program2)

    # print(program_sig1, program_sig2)
    sim = editdistance.eval(program_sig1, program_sig2)
    return sim


def _compute_consistency_score(env_name, hyp_program, nearest_neighbors, decode_results_dict, K=5):
    similar_questions = nearest_neighbors[env_name][:K]
    similar_questions = [q for q in similar_questions if q in decode_results_dict]

    support = 0.
    for nn_qid in similar_questions:
        similar_question_hyps = decode_results_dict[nn_qid]
        for nbr_hyp in similar_question_hyps:
            is_nbr_hyp_correct = nbr_hyp['is_correct']
            if is_nbr_hyp_correct:
                dist = compute_program_similarity(hyp_program, nbr_hyp['program'])
                similarity = dist * nbr_hyp['prob']
                support += similarity

    return support


def to_decode_results_dict(decode_results, test_envs):
    results = OrderedDict()
    for env, hyp_list in zip(test_envs, decode_results):
        results[env.name] = []
        for hyp in hyp_list:
            results[env.name].append(OrderedDict(
                # program=to_human_readable_program(hyp.trajectory.program, env),
                program=hyp.trajectory.program,
                is_correct=hyp.trajectory.reward == 1.,
                prob=hyp.prob
            ))
    return results


class Retrainer(object):
    def __init__(self, agent: PGAgent, train_set: List[QAProgrammingEnv], nearest_neighbors: Dict, config):
        self.agent = agent
        self.train_set = train_set
        self.config = config
        self.nearest_neighbors = nearest_neighbors

    def fine_tune(self):
        beam_size = self.config['beam_size']
        decoding_results = self.agent.decode_examples(self.train_set, beam_size=beam_size)
        decoding_results_dict = to_decode_results_dict(decoding_results, self.train_set)

        train_examples = []
        for env, hyp_list in zip(self.train_set, decoding_results):
            # hyp_list = [hyp for hyp in hyp_list if hyp.trajectory.reward == 1.]
            if not hyp_list:
                continue

            is_best_hyp_correct = hyp_list[0].trajectory.reward == 1.
            if not is_best_hyp_correct:
                # if True:
                correct_hyps = [hyp for hyp in hyp_list if hyp.trajectory.reward == 1.]
                if not correct_hyps: continue

                hyp_supports = [_compute_consistency_score(env_name=env.name,
                                                           hyp_program=hyp.trajectory.program,
                                                           nearest_neighbors=self.nearest_neighbors,
                                                           decode_results_dict=decoding_results,
                                                           K=3) for hyp in correct_hyps]
                best_hyp_idx = np.argmax(hyp_supports)
                best_hyp = hyp_list[best_hyp_idx]
                train_examples.append(best_hyp.trajectory)
            else:
                train_examples.append(hyp_list[0].trajectory)

        print(f'Num. fine tune examples: {len(train_examples)}', file=sys.stderr)
        max_epoch = 1

        model = self.agent.train()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)

        for epoch in range(max_epoch):
            batch_iter = nn_util.batch_iter(train_examples,
                                            batch_size=32,
                                            shuffle=True)

            for batch_id, train_trajectories in enumerate(batch_iter):
                optimizer.zero_grad()

                # (batch_size)
                batch_log_prob = self.agent(train_trajectories)
                loss = -batch_log_prob.mean()

                loss.backward()
                loss_val = loss.item()

                # clip gradient
                grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.)

                optimizer.step()
