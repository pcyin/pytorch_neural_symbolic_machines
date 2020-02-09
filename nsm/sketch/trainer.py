import sys
from typing import Dict, List

import torch
from pytorch_pretrained_bert import BertAdam

from nsm.env_factory import Trajectory
from nsm.sketch.sketch import Sketch
from nsm.sketch.sketch_predictor import SketchPredictor


class SketchPredictorTrainer(object):
    def __init__(self, model: SketchPredictor, num_train_step: int, freeze_bert_for_niter: int, config: Dict):
        self.model = model

        bert_params = list([
            (p_name, p)
            for (p_name, p) in model.encoder_model.bert_model.named_parameters()
            if p.requires_grad])
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        bert_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.other_params = [
            p
            for n, p
            in model.named_parameters()
            if 'bert_model' not in n and p.requires_grad
        ]

        self.bert_optimizer = BertAdam(
            bert_grouped_parameters,
            lr=config['bert_learning_rate'],
            warmup=0.1,
            t_total=num_train_step
        )

        self.optimizer = torch.optim.Adam(
            self.other_params,
            lr=0.001
        )

        self.freeze_bert_for_niter = freeze_bert_for_niter

    def step(self, trajectories: List[Trajectory], train_iter: int, context_encoding: Dict = None):
        sketches = [
            Sketch(traj.program)
            for traj in trajectories
        ]

        self.bert_optimizer.zero_grad()
        self.optimizer.zero_grad()

        sketch_log_prob = self.model(
            [
                traj.context for traj in trajectories
            ],
            sketches
        )
        sketch_loss = -sketch_log_prob.mean()
        sketch_loss.backward()

        if train_iter % 10 == 0:
            print(f'[SketchManagerTrainer] loss={sketch_loss.item()}', file=sys.stderr)

        torch.nn.utils.clip_grad_norm_(self.other_params, 5.)

        self.optimizer.step()
        if train_iter > self.freeze_bert_for_niter:
            self.bert_optimizer.step()
        elif train_iter == self.freeze_bert_for_niter:
            self.optimizer = torch.optim.Adam(self.other_params, lr=0.001)