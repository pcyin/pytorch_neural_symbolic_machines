import sys
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForPreTraining
from pytorch_pretrained_bert import BertAdam
from typing import List, Tuple, Any, Dict

from nsm import nn_util
from nsm.env_factory import Environment, Trajectory
from nsm.executor_factory import SimpleKGExecutor, TableExecutor
from nsm.sketch.sketch import Sketch

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


def get_executor_api():
    dummy_kg = {
        'kg': None,
        'num_props': [],
        'datetime_props': [],
        'props': [],
        'row_ents': []
    }

    api = TableExecutor(dummy_kg).get_api()

    return api


class TrainableSketchManager(nn.Module):
    def __init__(
        self,
        bert_model: BertPreTrainedModel,
        tokenizer: BertTokenizer,
        hidden_size: int
    ):
        nn.Module.__init__(self)

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.src_encoding_size = bert_model.config.hidden_size
        self.hidden_size = hidden_size

        self.attention_value_to_key = nn.Linear(self.src_encoding_size, hidden_size, bias=False)

        token_embed_size = hidden_size
        self.decoder_lstm = nn.LSTMCell(
            hidden_size + token_embed_size,
            hidden_size
        )
        self.att_vec_linear = nn.Linear(hidden_size + self.src_encoding_size, self.hidden_size, bias=False)

        self.executor_api = get_executor_api()
        operators = sorted(self.executor_api['func_dict'])
        self.sketch_vocab = {
            token: idx
            for idx, token
            in enumerate(['<s>', '</s>'] + operators)
        }
        self.sketch_id2token = {
            idx: token
            for token, idx
            in self.sketch_vocab.items()
        }

        self.sketch_token_embedding = nn.Embedding(
            len(self.sketch_vocab),
            token_embed_size
        )

        self.decoder_init_linear = nn.Linear(self.src_encoding_size, self.hidden_size)

        self.readout = nn.Linear(
            hidden_size,
            len(self.sketch_vocab),
            bias=False
        )

        self.dropout = nn.Dropout(0.2)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config):
        params = cls.default_params()
        params.update(config)

        bert_model = BertModel.from_pretrained(params['bert_model'])
        model = cls(
            bert_model,
            BertTokenizer.from_pretrained(params['bert_model']),
            params['sketch_decoder_hidden_size'],
        )

        return model

    @classmethod
    def default_params(cls):
        return {
            'bert_model': 'bert-base-uncased',
            'sketch_decoder_hidden_size': 256
        }

    def step(
        self, x: torch.Tensor,
        h_tm1: Tuple[torch.Tensor, torch.Tensor],
        att_values: torch.Tensor,
        att_keys: torch.Tensor,
        att_masks: torch.Tensor = None) -> Tuple[Tuple, torch.Tensor]:

        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = nn_util.dot_prod_attention(
            query=h_t,
            keys=att_keys,
            values=att_values,
            entry_masks=att_masks
        )

        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t

    def get_bert_input(self, questions: List[Any]):
        batch_size = len(questions)
        batch_token_ids = [
            self.tokenizer.convert_tokens_to_ids(['[CLS]'] + example + ['[SEP]'])
            for example
            in questions
        ]

        max_sequence_len = max(len(x) for x in batch_token_ids)
        input_array = np.zeros((batch_size, max_sequence_len), dtype=np.int64)
        mask_array = np.zeros((batch_size, max_sequence_len), dtype=np.float32)
        segment_array = np.zeros((batch_size, max_sequence_len), dtype=np.int64)

        for i in range(batch_size):
            seq_len = len(batch_token_ids[i])
            input_array[i, :seq_len] = batch_token_ids[i]
            mask_array[i, :seq_len] = 1

        bert_input = {
            'input_ids': torch.tensor(input_array, device=self.device),
            'segment_ids': torch.tensor(segment_array, device=self.device),
            'attention_mask': torch.tensor(mask_array, device=self.device)
        }

        return bert_input

    def to_tensor_dict(self, examples: List[Any], sketches: List[Sketch]):
        batch_size = len(examples)

        max_sketch_len = max(2 + len(sketch.operators) for sketch in sketches)
        sketch_token_ids = np.zeros((batch_size, max_sketch_len), dtype=np.int64)
        sketch_mask = np.zeros((batch_size, max_sketch_len), dtype=np.float32)

        for i, (example, sketch) in enumerate(
                zip(examples, sketches)):

            sketch = sketches[i]
            sketch_token_id = [
                self.sketch_vocab[token]
                for token
                in ['<s>'] + sketch.operators + ['</s>']
            ]

            sketch_token_ids[i, :len(sketch_token_id)] = sketch_token_id
            sketch_mask[i, :len(sketch_token_id)] = 1.

        bert_input = self.get_bert_input(examples)
        tensor_dict = {
            'bert_input': bert_input,
            'tgt_sketch_token_ids': torch.tensor(sketch_token_ids, device=self.device),
            'tgt_mask': torch.tensor(sketch_mask, device=self.device)
        }

        tensor_dict['src_mask'] = tensor_dict['bert_input']['attention_mask']

        return tensor_dict

    def forward(self, examples: List[Any], sketches: List[Sketch]):
        tensor_dict = self.to_tensor_dict(examples, sketches)

        # (batch_size, sequence_len, encoding_size)
        src_encodings = self.encode(**tensor_dict['bert_input'])

        decoder_init_vec = self.decoder_init(src_encodings)

        tgt_sketch_token_ids = tensor_dict['tgt_sketch_token_ids']

        # (batch_size, sketch_len - 1, )
        att_vecs = self.decode(
            src_encodings,
            tensor_dict['src_mask'],
            decoder_init_vec,
            tgt_sketch_token_ids[:, :-1]
        )

        # (batch_size, sketch_len, tgt_vocab_size)
        sketch_token_prob = torch.log_softmax(
            self.readout(att_vecs),
            dim=-1
        )

        # (batch_size, sketch_len)
        tgt_sketch_token_prob = torch.gather(
            sketch_token_prob,
            index=tgt_sketch_token_ids[:, 1:].unsqueeze(-1),
            dim=-1,
        ).squeeze(-1) * tensor_dict['tgt_mask'][:, 1:]

        sketch_prob = tgt_sketch_token_prob.sum(dim=-1)

        return sketch_prob

    def encode(self, input_ids, segment_ids, attention_mask):
        sequence_output, _ = self.bert_model(
            input_ids,
            segment_ids,
            attention_mask,
            output_all_encoded_layers=False
        )

        return sequence_output

    def decoder_init(self, src_encodings: torch.Tensor) -> torch.Tensor:
        cell_0 = self.decoder_init_linear(
            src_encodings[:, 0])

        state_0 = torch.tanh(cell_0)

        return [state_0, cell_0]

    def decode(
        self,
        src_encodings: torch.Tensor,
        src_mask: torch.Tensor,
        decoder_init_vec: torch.Tensor,
        tgt_token_ids: torch.Tensor
    ):
        batch_size = src_encodings.size(0)

        # (batch_size, )
        att_tm1 = src_encodings.new_zeros(batch_size, self.hidden_size)

        src_encodings_att_linear = self.attention_value_to_key(src_encodings)
        h_tm1 = decoder_init_vec
        tgt_token_embeds = self.sketch_token_embedding(tgt_token_ids)
        att_ves = []
        for y_tm1_embed in tgt_token_embeds.split(split_size=1, dim=1):
            y_tm1_embed = y_tm1_embed.squeeze(1)
            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t = self.step(
                x,
                h_tm1,
                src_encodings,
                src_encodings_att_linear,
                src_mask
            )

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (batch_size, tgt_seq_len, hidden_size)
        att_ves = torch.stack(att_ves, dim=1)

        return att_ves

    def get_sketches(self, question: List[Any], K=5):
        return self._beam_search(question, beam_size=K)

    def _beam_search(self, question, beam_size: int, max_decoding_time_step: int = 6) -> List[Sketch]:
        input_tensors = self.get_bert_input([question])
        # (batch_size, sequence_len, encoding_size)
        src_encodings = self.encode(**input_tensors)
        src_encodings_att_linear = self.attention_value_to_key(src_encodings)

        # (batch_size, hidden_size)
        att_tm1 = src_encodings.new_zeros(1, self.hidden_size)

        h_tm1 = self.decoder_init(src_encodings)

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num, -1, -1)
            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num, -1, -1)

            y_tm1 = torch.tensor(
                [self.sketch_vocab[hyp[-1]] for hyp in hypotheses],
                dtype=torch.long,
                device=self.device)
            y_tm1_embed = self.sketch_token_embedding(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t = self.step(
                x,
                h_tm1,
                exp_src_encodings,
                exp_src_encodings_att_linear,
                # input_tensors['attention_mask']
            )

            log_p_t = torch.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.sketch_vocab)
            hyp_word_ids = top_cand_hyp_pos % len(self.sketch_vocab)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.sketch_id2token[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    if len(new_hyp_sent) > 2:  # <s> CONTENT </s>
                        completed_hypotheses.append(Hypothesis(value=new_hyp_sent,
                                                               score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        sketches = [
            self.hypothesis_to_sketch(hyp)
            for hyp
            in completed_hypotheses
        ]

        return sketches

    def hypothesis_to_sketch(self, hyp: Hypothesis) -> Sketch:
        sketch_tokens = []
        for token in hyp.value:
            if token == '<s>':
                continue
            elif token in self.executor_api['func_dict']:
                op = self.executor_api['func_dict'][token]
                sketch_tokens.extend(['(', token] + ['v'] * len(op['args']) + [')'])
            elif token == '</s>':
                sketch_tokens.append('<END>')
            else:
                raise ValueError(f'Unknown token {token}')

        sketch = Sketch(sketch_tokens)

        return sketch


class SketchManagerTrainer(object):
    def __init__(self, model: TrainableSketchManager, num_train_step: int, config: Dict):
        self.model = model

        no_grad = ['pooler']
        bert_params = list([
            (p_name, p)
            for (p_name, p) in model.bert_model.named_parameters()
            if not any(pn in p_name for pn in no_grad)])
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
            t_total=num_train_step)

        self.optimizer = torch.optim.Adam(
            self.other_params,
            lr=0.001)

    def step(self, trajectories: List[Trajectory], train_iter: int):
        questions = [
            traj.context['question_tokens']
            for traj in trajectories
        ]

        sketches = [
            Sketch(traj.program)
            for traj in trajectories
        ]

        self.bert_optimizer.zero_grad()
        self.optimizer.zero_grad()

        sketch_log_prob = self.model(questions, sketches)
        sketch_loss = -sketch_log_prob.mean()
        sketch_loss.backward()

        if train_iter % 10 == 0:
            print(f'[SketchManagerTrainer] loss={sketch_loss.item()}', file=sys.stderr)

        torch.nn.utils.clip_grad_norm_(self.other_params, 5.)
        self.bert_optimizer.step()
        self.optimizer.step()
