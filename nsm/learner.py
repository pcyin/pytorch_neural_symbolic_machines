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
from nsm.evaluator import Evaluation
from nsm.program_cache import SharedProgramCache
from nsm.retrainer import Retrainer, load_nearest_neighbors

import torch
from tensorboardX import SummaryWriter

from nsm.dist_util import STOP_SIGNAL


class Learner(Process):
    def __init__(self, config, gpu_id=-1, shared_program_cache: SharedProgramCache = None):
        super(Learner, self).__init__(daemon=True)

        self.train_queue = multiprocessing.Queue()
        self.checkpoint_queue = multiprocessing.Queue()
        self.config = config
        self.gpu_id = gpu_id
        self.actor_message_vars = []
        self.current_model_path = None
        self.shared_program_cache = shared_program_cache

        self.actor_num = 0

    def run(self):
        # create agent
        self.agent = PGAgent.build(self.config).train()
        if self.gpu_id >= 0:
            self.agent = self.agent.to(torch.device("cuda", self.gpu_id))

        self.train()

    def train(self):
        model = self.agent
        train_iter = 0
        save_every_niter = self.config['save_every_niter']
        method = self.config['method']
        entropy_reg_weight = self.config['entropy_reg_weight']
        summary_writer = SummaryWriter(os.path.join(self.config['work_dir'], 'tb_log/train'))
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.001)
        config = self.config
        use_finetune = config['use_finetune']

        cum_loss = cum_examples = 0.
        t1 = time.time()

        if use_finetune:
            from table.experiments import load_environments
            print(f'load training and dev files for fine-tuning', file=sys.stderr)
            train_set = load_environments([config['train_file']],
                                          table_file=config['table_file'],
                                          vocab_file=config['vocab_file'],
                                          en_vocab_file=config['en_vocab_file'],
                                          embedding_file=config['embedding_file'])

            dev_set = load_environments([config['dev_file']],
                                        table_file=config['table_file'],
                                        vocab_file=config['vocab_file'],
                                        en_vocab_file=config['en_vocab_file'],
                                        embedding_file=config['embedding_file'])

            fine_tune_start = config['fine_tune_start']
            fine_tune_every_niter = config['fine_tune_every_niter']

            for env in dev_set:
                env.punish_extra_work = False
                env.use_cache = False

            print(f'load nearest neighbors', file=sys.stderr)
            nearest_neighbors = load_nearest_neighbors(config['nearest_neighbors_file'])

            retrainer = Retrainer(model, train_set, nearest_neighbors, config)

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

        max_train_step = config['max_train_step']
        while train_iter < max_train_step:
            train_iter += 1
            optimizer.zero_grad()

            train_samples, samples_info = self.train_queue.get()
            try:
                queue_size = self.train_queue.qsize()
                # print(f'[Learner] train_iter={train_iter} train queue size={queue_size}', file=sys.stderr)
                summary_writer.add_scalar('train_queue_size', queue_size, train_iter)
            except NotImplementedError:
                pass

            train_trajectories = [sample.trajectory for sample in train_samples]

            # (batch_size)
            batch_log_prob, entropy = self.agent(train_trajectories, entropy=True)

            train_sample_weights = batch_log_prob.new_tensor([s.weight for s in train_samples])
            batch_log_prob = batch_log_prob * train_sample_weights

            loss = -batch_log_prob.mean()
            # loss = -batch_log_prob.sum() / max_batch_size

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
            if 'clip_frac' in samples_info:
                summary_writer.add_scalar('sample_clip_frac', samples_info['clip_frac'], train_iter)

            cum_loss += loss_val * len(train_samples)
            cum_examples += len(train_samples)

            if use_finetune and train_iter >= fine_tune_start and train_iter % fine_tune_every_niter == 0:
                print(f'[FineTune] start iterative fine tuning...', file=sys.stderr)
                dev_scores = []

                dev_result = Evaluation.evaluate(model, dev_set, beam_size=config['beam_size'])
                dev_score = dev_result['accuracy']

                print(f'[FineTune] initial dev score={dev_score}', file=sys.stderr)
                dev_scores.append(dev_score)
                model_state = model.state_dict()
                model_state_file = os.path.join(config['work_dir'], 'model.tmp.state.bin')
                torch.save(model_state, model_state_file)

                for i in range(config['fine_tune_nepoch']):
                    retrainer.fine_tune()

                    dev_result = Evaluation.evaluate(model, dev_set, beam_size=config['beam_size'])
                    dev_score = dev_result['accuracy']

                    print(f'[FineTune] Epoch {i} dev score={dev_score}', file=sys.stderr)
                    sys.stderr.flush()
                    if dev_score > max(dev_scores):
                        print(f'save current best model', file=sys.stderr)
                        model_state = model.state_dict()
                        torch.save(model_state, model_state_file)

                    dev_scores.append(dev_score)

                # if np.argmax(dev_scores) > 0:
                print(f'[FineTune] reload best model {np.argmax(dev_scores)}', file=sys.stderr)
                model_state = torch.load(model_state_file)
                model.load_state_dict(model_state)

                model.train()

            if train_iter % save_every_niter == 0:
                print(f'[Learner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
                      f'{cum_examples} examples ({cum_examples / (time.time() - t1)} examples/s)', file=sys.stderr)
                cum_loss = cum_examples = 0.
                t1 = time.time()

                self.update_model_to_actors(train_iter)
            else:
                self.push_new_model(self.current_model_path)

        for i in range(self.actor_num):
            self.checkpoint_queue.put(STOP_SIGNAL)
        self.eval_msg_val.value = STOP_SIGNAL.encode()

    def update_model_to_actors(self, train_iter):
        t1 = time.time()
        model_state = self.agent.state_dict()
        model_save_path = os.path.join(self.config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
        torch.save(model_state, model_save_path)

        self.push_new_model(model_save_path)
        print(f'[Learner] pushed model [{model_save_path}] (took {time.time() - t1}s)', file=sys.stderr)

        if self.current_model_path:
            os.remove(self.current_model_path)
        self.current_model_path = model_save_path

    def push_new_model(self, model_path):
        self.checkpoint_queue.put(model_path)
        if model_path:
            self.eval_msg_val.value = model_path.encode()

    def register_actor(self, actor):
        actor.checkpoint_queue = self.checkpoint_queue
        actor.train_queue = self.train_queue
        self.actor_num += 1

    def register_evaluator(self, evaluator):
        msg_var = multiprocessing.Array(ctypes.c_char, 4096)
        self.eval_msg_val = msg_var
        evaluator.message_var = msg_var
