import ctypes
import heapq
import os
import random
import time
from typing import List

import numpy as np
import sys

import multiprocessing
from multiprocessing import Process

from nsm import nn_util
from nsm.agent_factory import PGAgent, Sample
from nsm.env_factory import Trajectory

import torch


class Learner(Process):
    def __init__(self, config):
        super(Learner, self).__init__(daemon=True)

        self.train_queue = multiprocessing.Queue()
        self.config = config
        self.actor_message_vars = []

    def run(self):
        # create agent
        self.agent = PGAgent.build(self.config)

        self.train()

    def train(self):
        model = self.agent
        train_iter = 0
        save_every_niter = self.config['save_every_niter']
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)

        nn_util.glorot_init(params)

        while True:
            train_iter += 1
            optimizer.zero_grad()

            train_samples = self.train_queue.get()
            train_trajectories = [sample.trajectory for sample in train_samples]

            # (batch_size)
            batch_loss = self.agent(train_trajectories)

            loss = -batch_loss.mean()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(params, 5.)

            optimizer.step()

            print(f'[Learner] train_iter={train_iter} loss={loss.item()}', file=sys.stderr)
            del loss

            if train_iter > 0 and train_iter % save_every_niter == 0:
                model_state = model.state_dict()
                model_save_path = os.path.join(self.config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
                torch.save(model_state, model_save_path)

                self.push_new_model(model_save_path)

    def push_new_model(self, model_path):
        t1 = time.time()
        for msg_val in self.actor_message_vars:
            msg_val.value = model_path.encode()

        t2 = time.time()

        print('[Learner] pushed model [%s] (took %d s)' % (model_path, t2 - t1), file=sys.stderr)

    def register_actor(self, actor):
        msg_var = multiprocessing.Array(ctypes.c_char, 4096)
        self.actor_message_vars.append(msg_var)

        actor.message_var = msg_var
        actor.train_queue = self.train_queue

    def register_evaluator(self, evaluator):
        msg_var = multiprocessing.Array(ctypes.c_char, 4096)
        self.actor_message_vars.append(msg_var)

        evaluator.message_var = msg_var
