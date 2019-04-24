import os
import re
import sys
import time
from typing import List, Union
import numpy as np
from tensorboardX import SummaryWriter

from nsm.agent_factory import Sample, PGAgent
from nsm.env_factory import QAProgrammingEnv

from multiprocessing import Queue, Process

import torch


class Evaluation(object):
    @staticmethod
    def evaluate(model: PGAgent, dataset: List[QAProgrammingEnv], beam_size: int):
        was_training = model.training
        model.eval()

        decode_results = model.decode_examples(dataset, beam_size=beam_size)
        eval_results = Evaluation.evaluate_decode_results(dataset, decode_results)

        if was_training:
            model.train()

        return eval_results

    @staticmethod
    def evaluate_decode_results(dataset: List[QAProgrammingEnv], decoding_results=Union[List[List[Sample]], List[Sample]], verbose=False):
        if isinstance(decoding_results[0], Sample):
            decoding_results = [[hyp] for hyp in decoding_results]

        acc_list = []
        oracle_acc_list = []

        for env, hyp_list in zip(dataset, decoding_results):
            is_top_correct = len(hyp_list) > 0 and hyp_list[0].trajectory.reward == 1.
            has_correct_program = any(hyp.trajectory.reward == 1. for hyp in hyp_list)

            acc_list.append(is_top_correct)
            oracle_acc_list.append(has_correct_program)

        eval_result = dict(accuracy=np.average(acc_list),
                           oracle_accuracy=np.average(oracle_acc_list))

        return eval_result


class Evaluator(Process):
    def __init__(self, config, eval_file, gpu_id=-1):
        super(Evaluator, self).__init__()
        self.eval_queue = Queue()
        self.config = config
        self.eval_file = eval_file
        self.gpu_id = gpu_id

        self.model_path = 'INIT_MODEL'
        self.message_var = None

    def run(self):
        self.agent = PGAgent.build(self.config).eval()
        if self.gpu_id >= 0:
            self.agent = self.agent.to(torch.device("cuda", self.gpu_id))

        self.load_environments()
        summary_writer = SummaryWriter(os.path.join(self.config['work_dir'], 'tb_log/dev'))

        dev_scores = []

        while True:
            if self.check_and_load_new_model():
                print(f'[Evaluator] evaluate model [{self.model_path}]', file=sys.stderr)
                t1 = time.time()

                decode_results = self.agent.decode_examples(self.environments, beam_size=self.config['beam_size'])

                eval_results = Evaluation.evaluate_decode_results(self.environments, decode_results)

                t2 = time.time()
                print(f'[Evaluator] result={repr(eval_results)}, took {t2 - t1}s', file=sys.stderr)

                summary_writer.add_scalar('eval/accuracy', eval_results['accuracy'], self.get_global_step())
                summary_writer.add_scalar('eval/oracle_accuracy', eval_results['oracle_accuracy'], self.get_global_step())

                dev_score = eval_results['accuracy']
                if not dev_scores or max(dev_scores) < dev_score:
                    print(f'[Evaluator] save the current best model', file=sys.stderr)
                    self.agent.save(os.path.join(self.config['work_dir'], 'model.best.bin'))

                dev_scores.append(dev_score)

                sys.stderr.flush()

            time.sleep(2)  # in seconds

    def load_environments(self):
        from table.experiments import load_environments
        envs = load_environments([self.eval_file],
                                 table_file=self.config['table_file'],
                                 vocab_file=self.config['vocab_file'],
                                 en_vocab_file=self.config['en_vocab_file'],
                                 embedding_file=self.config['embedding_file'])
        for env in envs:
            env.use_cache = False
            env.punish_extra_work = False

        self.environments = envs

    def check_and_load_new_model(self):
        new_model_path = self.message_var.value.decode()
        # print(f'[Evaluator] current newest model path [{new_model_path}]', file=sys.stderr)
        if new_model_path and new_model_path != self.model_path:
            t1 = time.time()

            state_dict = torch.load(new_model_path, map_location=lambda storage, loc: storage)
            self.agent.load_state_dict(state_dict)
            self.model_path = new_model_path

            t2 = time.time()
            print('[Evaluator] loaded new model [%s] (took %.2f s)' % (new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False

    def get_global_step(self):
        train_iter = re.search('iter(\d+)?', self.model_path).group(1)

        return int(train_iter)
