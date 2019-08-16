import ctypes
import heapq
import json
import os
import random
import time
from itertools import chain
from pathlib import Path
from typing import List, Dict

import numpy as np
import sys

import torch.multiprocessing as torch_mp
import multiprocessing

from pytorch_pretrained_bert import BertAdam

from nsm import nn_util
from nsm.agent_factory import PGAgent
from nsm.consistency_utils import ConsistencyModel, QuestionSimilarityModel
from nsm.retrainer import Retrainer, load_nearest_neighbors
from nsm.evaluator import Evaluation
from nsm.program_cache import SharedProgramCache

import torch
from tensorboardX import SummaryWriter

from nsm.dist_util import STOP_SIGNAL
from nsm.sketch.sketch_generator import TrainableSketchManager, SketchManagerTrainer


class Learner(torch_mp.Process):
    def __init__(self, config: Dict, device: torch.device, shared_program_cache: SharedProgramCache = None):
        super(Learner, self).__init__(daemon=True)

        self.train_queue = multiprocessing.Queue()
        self.checkpoint_queue = multiprocessing.Queue()
        self.config = config
        self.device = device
        self.actor_message_vars = []
        self.current_model_path = None
        self.shared_program_cache = shared_program_cache

        self.actor_num = 0

    def run(self):
        # initialize cuda context
        self.device = torch.device(self.device)

        if 'cuda' in self.device.type:
            torch.cuda.set_device(self.device)

        # seed the random number generators
        nn_util.init_random_seed(self.config['seed'], self.device)

        self.agent = PGAgent.build(self.config).to(self.device).train()

        self.train()

    def train(self):
        model = self.agent
        config = self.config
        work_dir = Path(config['work_dir'])
        train_iter = 0
        save_every_niter = config['save_every_niter']
        entropy_reg_weight = config['entropy_reg_weight']
        summary_writer = SummaryWriter(os.path.join(config['work_dir'], 'tb_log/train'))
        max_train_step = config['max_train_step']
        save_program_cache_niter = config.get('save_program_cache_niter', 0)
        freeze_bert_for_niter = config.get('freeze_bert_niter', 0)
        # if freeze_bert:
        #     for p in model.encoder.bert_model.parameters():
        #         p.requires_grad = False

        bert_params = [
            (p_name, p)
            for (p_name, p) in model.named_parameters()
            if 'bert_model' in p_name
        ]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        num_train_optimization_steps = max_train_step

        bert_optimizer = BertAdam(
            bert_grouped_parameters,
            lr=self.config['bert_learning_rate'],
            warmup=0.1,
            t_total=num_train_optimization_steps)

        other_params = [
            p
            for n, p
            in model.named_parameters()
            if 'bert_model' not in n and p.requires_grad
        ]
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

        use_trainable_sketch_manager = config.get('use_trainable_sketch_manager', False)
        optimizer = torch.optim.Adam(other_params, lr=0.001)

        max_batch_size = self.config['batch_size'] * (self.config['n_replay_samples'] + self.config['n_policy_samples'])

        # nn_util.glorot_init(params)
        # torch.nn.init.zeros_(model.decoder.output_feature_linear.weight)
        # torch.nn.init.normal_(model.encoder.context_embedder.trainable_embedding.weight, mean=0., std=0.1)
        # torch.nn.init.normal_(model.decoder.builtin_func_embeddings.weight, mean=0., std=0.1)

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
            batch_log_prob, meta_info = self.agent(train_trajectories, return_info=True)

            train_sample_weights = batch_log_prob.new_tensor([s.weight for s in train_samples])
            batch_log_prob = batch_log_prob * train_sample_weights

            loss = -batch_log_prob.mean()
            summary_writer.add_scalar('parser_loss', loss.item(), train_iter)
            # loss = -batch_log_prob.sum() / max_batch_size

            if entropy_reg_weight != 0.:
                entropy = entropy.mean()
                ent_reg_loss = - entropy_reg_weight * entropy  # maximize entropy
                loss = loss + ent_reg_loss

                summary_writer.add_scalar('entropy', entropy.item(), train_iter)
                summary_writer.add_scalar('entropy_reg_loss', ent_reg_loss.item(), train_iter)

            if use_trainable_sketch_manager:
                context_encoding = meta_info['context_encoding']['table_bert_encoding'] if \
                    config.get('sketch_decoder_use_table_bert', False) and \
                    config.get('sketch_decoder_use_parser_table_bert', False) \
                    else None

                sketch_log_prob = model.sketch_manager.get_trajectory_sketch_prob(
                    train_trajectories,
                    context_encoding=context_encoding
                )

                sketch_loss = -(sketch_log_prob * train_sample_weights).mean()
                summary_writer.add_scalar('sketch_loss', sketch_loss.item(), train_iter)

                loss = loss + sketch_loss

            loss.backward()
            loss_val = loss.item()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(other_params, 5.)

            optimizer.step()

            if train_iter > freeze_bert_for_niter:
                bert_optimizer.step()
            elif train_iter == freeze_bert_for_niter:
                print(f'[Learner] train_iter={train_iter} reset Adam optimizer and start fine-tuning BERT')
                optimizer = torch.optim.Adam(other_params, lr=0.001)

            bert_optimizer.zero_grad()

            # print(f'[Learner] train_iter={train_iter} loss={loss_val}', file=sys.stderr)
            del loss
            if entropy_reg_weight != 0.: del entropy

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

                # log stats of the program cache
                program_cache_stat = self.shared_program_cache.stat()
                summary_writer.add_scalar(
                    'avg_num_programs_in_cache',
                    program_cache_stat['num_entries'] / program_cache_stat['num_envs'],
                    train_iter
                )
                summary_writer.add_scalar(
                    'num_programs_in_cache',
                    program_cache_stat['num_entries'],
                    train_iter
                )
            else:
                self.push_new_model(self.current_model_path)

            if save_program_cache_niter > 0 and train_iter % save_program_cache_niter == 0:
                program_cache_file = work_dir / 'log' / f'program_cache.iter{train_iter}.json'
                program_cache = self.shared_program_cache.all_programs()
                json.dump(
                    program_cache,
                    program_cache_file.open('w'),
                    indent=2
                )
        # for i in range(self.actor_num):
        #     self.checkpoint_queue.put(STOP_SIGNAL)
        # self.eval_msg_val.value = STOP_SIGNAL.encode()

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
