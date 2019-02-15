import ctypes
import heapq
import os
import random
import time
from itertools import chain
from typing import List

import numpy as np
import sys

import multiprocessing
from multiprocessing import Process

from nsm import nn_util
from nsm.agent_factory import PGAgent, Sample
from nsm.env_factory import Trajectory

import torch
from tensorboardX import SummaryWriter


class Learner(Process):
    def __init__(self, config, gpu_id=-1, summary_writer=None):
        super(Learner, self).__init__(daemon=True)

        self.train_queue = multiprocessing.Queue()
        self.config = config
        self.gpu_id = gpu_id
        self.actor_message_vars = []

    def run(self):
        # create agent
        self.agent = PGAgent.build(self.config).train()
        if self.gpu_id >= 0:
            self.agent.to(torch.device("cuda:%d" % self.gpu_id))

        self.train()

    def train(self):
        model = self.agent
        train_iter = 0
        save_every_niter = self.config['save_every_niter']
        entropy_reg_weight = self.config['entropy_reg_weight']
        summary_writer = SummaryWriter(os.path.join(self.config['work_dir'], 'tb_log/train'))
        old_model_path = None
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)

        cum_loss = cum_examples = 0.
        t1 = time.time()

        max_batch_size = self.config['batch_size'] * (self.config['n_replay_samples'] + self.config['n_policy_samples'])

        nn_util.glorot_init(params)
        torch.nn.init.zeros_(model.decoder.output_feature_linear.weight)
        torch.nn.init.normal_(model.encoder.context_embedder.trainable_embedding.weight, mean=0., std=0.1)
        torch.nn.init.normal_(model.decoder.builtin_func_embeddings.weight, mean=0., std=0.1)

        # set forget gate bias to 1, as in tensorflow
        for name, p in chain(model.decoder.rnn_cell.named_parameters(), model.encoder.lstm_encoder.named_parameters()):
            if 'bias' in name:
                n = p.size(0)
                forget_start_idx, forget_end_idx = n // 4, n // 2
                p.data[forget_start_idx:forget_end_idx].fill_(1.)

        while True:
            train_iter += 1
            optimizer.zero_grad()

            train_samples = self.train_queue.get()
            train_trajectories = [sample.trajectory for sample in train_samples]

            # (batch_size)
            batch_log_prob, entropy = self.agent(train_trajectories, entropy=True)

            train_sample_weights = batch_log_prob.new_tensor([s.prob for s in train_samples])
            batch_log_prob = batch_log_prob * train_sample_weights

            # loss = -batch_log_prob.mean()
            loss = -batch_log_prob.sum() / max_batch_size

            if entropy_reg_weight != 0.:
                entropy = entropy.mean()
                ent_reg_loss = - entropy_reg_weight * entropy  # maximize entropy
                loss = loss + ent_reg_loss

                summary_writer.add_scalar('entropy', entropy.item(), train_iter)
                summary_writer.add_scalar('entropy_reg_loss', ent_reg_loss.item(), train_iter)

            loss.backward()
            loss_val = loss.item()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 5.)

            optimizer.step()

            # print(f'[Learner] train_iter={train_iter} loss={loss_val}', file=sys.stderr)
            del loss
            if entropy_reg_weight != 0.: del entropy

            summary_writer.add_scalar('train_loss', loss_val, train_iter)
            cum_loss += loss_val * len(train_samples)
            cum_examples += len(train_samples)

            if train_iter > 0 and train_iter % save_every_niter == 0:
                print(f'[Learner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
                      f'{cum_examples} examples ({cum_examples / (time.time() - t1)} examples/s)', file=sys.stderr)
                cum_loss = cum_examples = 0.
                t1 = time.time()

                model_state = model.state_dict()
                model_save_path = os.path.join(self.config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
                torch.save(model_state, model_save_path)

                self.push_new_model(model_save_path)

                if old_model_path:
                    os.remove(old_model_path)
                old_model_path = model_save_path

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
