import sys
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.utils
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, \
    BertForPreTraining
from pytorch_pretrained_bert import BertAdam
from typing import List, Tuple, Any, Dict, Optional, Union

from nsm import nn_util
from nsm.env_factory import Environment, Trajectory
from nsm.executor_factory import SimpleKGExecutor, TableExecutor
from nsm.sketch.sketch import Sketch
from table.bert.data_model import Example
from table.bert.model import TableBERT

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
        hidden_size: int,
        token_embed_size: int,
        freeze_bert: bool = False,
        use_lstm_encoder: bool = False,
        dropout: float = 0.2
    ):
        nn.Module.__init__(self)

        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.freeze_bert = freeze_bert
        self.use_lstm_encoder = use_lstm_encoder
        self.use_table_bert = isinstance(bert_model, TableBERT)
        self.use_column_encoding = self.use_table_bert

        bert_output_size = bert_model.config.hidden_size

        if use_lstm_encoder:
            self.lstm_encoder = nn.LSTM(
                bert_output_size, hidden_size,
                bidirectional=True, batch_first=True
            )

            self.src_encoding_size = hidden_size * 2
        else:
            self.src_encoding_projection = nn.Linear(
                bert_output_size, hidden_size,
                bias=False
            )

            self.column_encoding_projection = nn.Linear(
                bert_output_size, hidden_size,
                bias=False
            )

            self.src_encoding_size = hidden_size

        self.src_attention_value_to_key = nn.Linear(
            self.src_encoding_size,
            hidden_size, bias=False
        )

        self.column_attention_value_to_key = nn.Linear(
            self.src_encoding_size,
            hidden_size, bias=False
        )

        self.decoder_init_linear = nn.Linear(
            self.src_encoding_size,
            self.hidden_size
        )

        self.decoder_lstm = nn.LSTMCell(
            hidden_size + token_embed_size,
            hidden_size
        )

        context_vec_size = self.src_encoding_size * 2 \
            if self.use_table_bert \
            else self.src_encoding_size

        self.decoder_att_vec_linear = nn.Linear(
            hidden_size + context_vec_size,
            self.hidden_size,
            bias=False
        )

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

        self.readout = nn.Linear(
            hidden_size,
            len(self.sketch_vocab),
            bias=False
        )

        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        print('Init sketch generator weights')

        def _init_weights(_module):
            initializer_range = self.bert_model.config.initializer_range

            if isinstance(_module, (nn.Linear, nn.Embedding)):
                _module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(_module, nn.Linear) and _module.bias is not None:
                _module.bias.data.zero_()

        for module in [
            self.src_encoding_projection, self.column_encoding_projection,
            self.src_attention_value_to_key, self.column_attention_value_to_key,
            self.decoder_lstm,
            self.decoder_att_vec_linear, self.decoder_init_linear,
            self.readout, self.sketch_token_embedding
        ]:
            module.apply(_init_weights)

        # for m_name, module in self.named_modules():
        #     if 'encoder_model' not in m_name:
        #         module.apply(_init_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config, **kwargs):
        params = cls.default_params()
        params.update(config)

        if params['sketch_decoder_use_table_bert']:
            bert_model = kwargs['encoder'].bert_model
            tokenizer = bert_model.tokenizer
        else:
            bert_model = BertModel.from_pretrained(params['bert_model'])
            tokenizer = BertTokenizer.from_pretrained(params['bert_model'])

        model = cls(
            bert_model,
            tokenizer,
            hidden_size=params['sketch_decoder_hidden_size'],
            token_embed_size=params['sketch_decoder_token_embed_size'],
            freeze_bert=params['sketch_decoder_freeze_bert'],
            use_lstm_encoder=params['sketch_decoder_use_lstm_encoder']
        )

        return model

    @classmethod
    def default_params(cls):
        return {
            'sketch_decoder_use_table_bert': False,
            'bert_model': 'bert-base-uncased',
            'sketch_decoder_hidden_size': 256,
            'sketch_decoder_token_embed_size': 256,
            'sketch_decoder_freeze_bert': False,
            'sketch_decoder_use_lstm_encoder': False
        }

    def step(
        self, x: torch.Tensor,
        h_tm1: Tuple[torch.Tensor, torch.Tensor],
        src_encodings: Dict,
        column_encodings: Dict
    ) -> Tuple[Tuple, torch.Tensor]:

        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        # (batch_size, question_len, encoding_size)
        ctx_q_t, alpha_q_t = nn_util.dot_prod_attention(
            query=h_t,
            keys=src_encodings['key'],
            values=src_encodings['value'],
            entry_masks=src_encodings['mask']
        )

        # (batch_size, column_num, encoding_size)
        ctx_column_t, alpha_column_t = nn_util.dot_prod_attention(
            query=h_t,
            keys=column_encodings['key'],
            values=column_encodings['value'],
            entry_masks=column_encodings['mask']
        )

        # (batch_size, context_vector_size)
        ctx_t = torch.cat([ctx_q_t, ctx_column_t], dim=-1)

        att_t = torch.tanh(self.decoder_att_vec_linear(torch.cat([h_t, ctx_t], 1)))
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

    def to_tensor_dict(self, examples: List[Any], sketches: List[Sketch] = None, context_encoding: Dict = None):
        batch_size = len(examples)
        tensor_dict = {}

        if context_encoding is None:
            bert_input = self.get_bert_input(examples)
            tensor_dict['bert_input'] = bert_input

        if sketches is not None:
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

            tensor_dict['tgt_sketch_token_ids'] = torch.tensor(sketch_token_ids, device=self.device)
            tensor_dict['tgt_mask'] = torch.tensor(sketch_mask, device=self.device)

        return tensor_dict

    def forward(self, examples: List[Any], sketches: List[Sketch], context_encoding: Dict = None):
        if self.use_table_bert:
            assert context_encoding is not None

        tensor_dict = self.to_tensor_dict(
            examples, sketches, context_encoding)

        # (batch_size, sequence_len, encoding_size)
        # (batch_size, max_column_len, encoding_size)
        src_encodings, column_encodings, decoder_init_state = self.encode(tensor_dict, context_encoding)

        tgt_sketch_token_ids = tensor_dict['tgt_sketch_token_ids']

        # (batch_size, sketch_len - 1, )
        att_vecs = self.decode(
            src_encodings,
            column_encodings,
            decoder_init_state,
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

    def get_trajectory_sketch_prob(self, trajectories: List[Trajectory], context_encoding: Dict = None):
        questions = [
            traj.context['question_tokens']
            for traj in trajectories
        ]

        sketches = [
            Sketch(traj.program)
            for traj in trajectories
        ]

        sketch_log_prob = self.forward(questions, sketches, context_encoding)
        # sketch_loss = -sketch_log_prob.mean()

        return sketch_log_prob

    def encode(self, tensor_dict: Dict, context_encoding: Dict = None):
        if self.use_table_bert:
            if self.freeze_bert:
                context_encoding['question_encoding'] = context_encoding['question_encoding'].detach()
                context_encoding['column_encoding'] = context_encoding['column_encoding'].detach()

            src_encoding_var = self.src_encoding_projection(
                context_encoding['question_encoding'])

            src_encodings = {
                'value': src_encoding_var,
                'key': self.src_attention_value_to_key(
                    src_encoding_var),
                'mask': context_encoding['question_token_mask']
            }

            column_encoding_var = self.column_encoding_projection(
                context_encoding['column_encoding'])

            column_encodings = {
                'value': column_encoding_var,
                'key': self.column_attention_value_to_key(
                    column_encoding_var),
                'mask': context_encoding['column_mask']
            }

            decoder_state_init_vec = src_encodings['value'][:, 0]  # encoding of the [CLS] label
        else:
            src_encodings, decoder_state_init_vec = self._encode(**tensor_dict)
            column_encodings = None

        decoder_init_state = self.decoder_init(decoder_state_init_vec)

        return src_encodings, column_encodings, decoder_init_state

    def _encode(self, input_ids, segment_ids, attention_mask):
        bert_sequence_output, _ = self.bert_model(
            input_ids,
            segment_ids,
            attention_mask,
            output_all_encoded_layers=False
        )

        if self.use_lstm_encoder:
            src_lens = [
                int(l)
                for l
                in attention_mask.sum(-1).cpu().tolist()
            ]
            packed_input = pack_padded_sequence(
                bert_sequence_output, src_lens,
                batch_first=True, enforce_sorted=False
            )

            src_encodings, (sorted_last_state, sorted_last_cell) = self.lstm_encoder(packed_input)
            src_encodings, _ = pad_packed_sequence(src_encodings, batch_first=True)

            # (num_directions, batch_size, hidden_size)
            last_cell = sorted_last_cell.index_select(1, packed_input.unsorted_indices)
            # (batch_size, num_directions * hidden_size)
            last_states = torch.cat([last_cell[0], last_cell[1]], dim=-1)

            # size = list(src_encodings.size())[:-1] + [2, -1]
            # # (batch_size, sequence_len, num_directions, hidden_size)
            # src_encodings_directional = src_encodings.view(*size)
            #
            # # (batch_size, sequence_len, hidden_size)
            # fwd_encodings = src_encodings_directional[:, :, 0, :]
            # bak_encoding = src_encodings_directional[:, :, 1, :]
            #
            # fwd_last_state = fwd_encodings[:, [l - 1 for l in src_lens], :]
            # bak_last_state = bak_encoding[:, 0, :]
            #
            # last_states = torch.cat([fwd_last_state, bak_last_state], dim=-1)
        else:
            src_encodings = bert_sequence_output
            last_states = bert_sequence_output[:, 0]

        decoder_init_state = self.decoder_init(src_encodings, last_states)

        return src_encodings, decoder_init_state

    def decoder_init(
        self,
        init_vec: torch.Tensor
    ) -> torch.Tensor:
        cell_0 = self.decoder_init_linear(init_vec)
        state_0 = torch.tanh(cell_0)

        return [state_0, cell_0]

    def decode(
        self,
        src_encodings: Dict,
        column_encodings: Dict,
        decoder_init_vec: torch.Tensor,
        tgt_token_ids: torch.Tensor
    ):
        batch_size = tgt_token_ids.size(0)

        # (batch_size, )
        att_tm1 = torch.zeros(batch_size, self.hidden_size, device=self.device)
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
                column_encodings
            )

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (batch_size, tgt_seq_len, hidden_size)
        att_ves = torch.stack(att_ves, dim=1)

        return att_ves

    def get_sketches(self, envs: List[Environment], K=5):
        # get context encoding from table BERT
        question_encoding, column_encoding, info = self.bert_model.encode([
            Example(question=e.context['question_tokens'], table=e.context['table'])
            for e in envs
        ])

        context_encoding = {
            'question_encoding': question_encoding,
            'column_encoding': column_encoding,
        }
        context_encoding.update(info['tensor_dict'])

        tensor_dict = self.to_tensor_dict(
            [
                env.get_context()['question_tokens']
                for env in envs
            ],
            sketches=None,
            context_encoding=context_encoding
        )

        src_encodings, column_encodings, decoder_init_state = self.encode(
            tensor_dict,
            context_encoding=context_encoding
        )

        return self.beam_search(envs, src_encodings, column_encodings, decoder_init_state, beam_size=K)

    def beam_search(
            self,
            examples: List[Any],
            src_encodings: Dict,
            column_encodings: Dict,
            decoder_init_state: Tuple[torch.Tensor, torch.Tensor],
            beam_size: int,
            max_decoding_time_step: int = 6
    ) -> List[List[Sketch]]:

        all_beams = [[['<s>']] for _ in examples]
        completed_hypotheses = [[] for _ in examples]

        hyp_question_ids = torch.tensor(list(range(len(examples))), dtype=torch.long, device=self.device)
        hyp_scores = torch.zeros(len(examples), dtype=torch.float, device=self.device)

        # (batch_size, hidden_size)
        att_tm1 = torch.zeros(len(examples), self.hidden_size, device=self.device)
        h_tm1 = decoder_init_state
        t = 0

        def __select_encodings(_encodings: Dict, indices: torch.Tensor):
            selected_tensor_dict = {
                name: value.index_select(0, indices)
                for name, value
                in _encodings.items()
            }

            return selected_tensor_dict

        while t < max_decoding_time_step:
            t += 1

            # (total_hyp_num, src_len, *)
            exp_src_encodings = __select_encodings(src_encodings, hyp_question_ids)
            exp_column_encodings = __select_encodings(column_encodings, hyp_question_ids)

            y_tm1 = torch.tensor(
                [self.sketch_vocab[hyp[-1]]
                 for beam in all_beams
                 for hyp in beam],
                dtype=torch.long,
                device=self.device)
            y_tm1_embed = self.sketch_token_embedding(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t = self.step(
                x,
                h_tm1,
                exp_src_encodings,
                exp_column_encodings
            )

            # (total_hyp_num, vocab_size)
            log_p_t = torch.log_softmax(self.readout(att_t), dim=-1)
            contiuating_hyp_candidate_scores = hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t

            # enumerate every beam
            new_hyp_question_ids = []
            new_hyp_scores = []
            new_hyp_prev_hyp_ids = []
            new_beams = []

            beam_hyp_begin = 0
            for q_idx, beam in enumerate(all_beams):
                cur_beam_size = len(beam)
                if cur_beam_size == 0:
                    new_beams.append([])
                    continue

                beam_hyp_end = beam_hyp_begin + cur_beam_size

                # (cur_beam_size)
                beam_contiuating_cand_scores = contiuating_hyp_candidate_scores[beam_hyp_begin: beam_hyp_end]
                live_hyp_num = beam_size - len(completed_hypotheses[q_idx])

                top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(beam_contiuating_cand_scores.view(-1), k=live_hyp_num)
                prev_hyp_ids = top_cand_hyp_pos / len(self.sketch_vocab)
                hyp_word_ids = top_cand_hyp_pos % len(self.sketch_vocab)

                new_beam = []
                for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                    prev_hyp_id = prev_hyp_id.item()
                    hyp_word_id = hyp_word_id.item()
                    cand_new_hyp_score = cand_new_hyp_score.item()

                    hyp_word = self.sketch_id2token[hyp_word_id]
                    new_hyp_sent = beam[prev_hyp_id] + [hyp_word]
                    if hyp_word == '</s>':
                        if len(new_hyp_sent) > 2:  # <s> CONTENT </s>
                            completed_hypotheses[q_idx].append(
                                Hypothesis(value=new_hyp_sent, score=cand_new_hyp_score))
                    else:
                        new_beam.append(new_hyp_sent)
                        new_hyp_prev_hyp_ids.append(prev_hyp_id + beam_hyp_begin)
                        new_hyp_question_ids.append(q_idx)
                        new_hyp_scores.append(cand_new_hyp_score)

                new_beams.append(new_beam)
                beam_hyp_begin = beam_hyp_end

            if all(len(beam) == 0 for beam in new_beams):
                break

            h_tm1 = (h_t[new_hyp_prev_hyp_ids], cell_t[new_hyp_prev_hyp_ids])
            att_tm1 = att_t[new_hyp_prev_hyp_ids]

            all_beams = new_beams
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)
            hyp_question_ids = torch.tensor(new_hyp_question_ids, dtype=torch.long, device=self.device)

        sketches = []
        for hyps in completed_hypotheses:
            hyps.sort(key=lambda hyp: hyp.score, reverse=True)

            question_sketches = [
                self.hypothesis_to_sketch(hyp)
                for hyp
                in hyps
            ]

            sketches.append(question_sketches)

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

        sketch = Sketch(sketch_tokens, prob=float(hyp.score))

        return sketch


class SketchManagerTrainer(object):
    def __init__(self, model: TrainableSketchManager, num_train_step: int, freeze_bert_for_niter: int, config: Dict):
        self.model = model

        no_grad = ['pooler']
        if self.model.use_table_bert:
            bert_params = no_decay = []
        else:
            bert_params = list([
                (p_name, p)
                for (p_name, p) in model.bert_model.named_parameters()
                if not any(pn in p_name for pn in no_grad) and p.requires_grad])
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

        bert_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # print(bert_grouped_parameters)

        self.other_params = [
            p
            for n, p
            in model.named_parameters()
            if 'encoder_model' not in n and p.requires_grad
        ]

        self.bert_optimizer = BertAdam(
            bert_grouped_parameters,
            lr=config['bert_learning_rate'],
            warmup=0.1,
            t_total=num_train_step)

        self.optimizer = torch.optim.Adam(
            self.other_params,
            lr=0.001)

        self.freeze_bert_for_niter = freeze_bert_for_niter

    def step(self, trajectories: List[Trajectory], train_iter: int, context_encoding: Dict = None):
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

        sketch_log_prob = self.model(questions, sketches, context_encoding)
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
