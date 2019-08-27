import json
import sys
from collections import namedtuple
from pathlib import Path

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
from nsm.parser_module.bert_encoder import BertEncoder
from nsm.sketch.sketch import Sketch
from table.bert.data_model import Example
from table.bert.model import TableBERT

SketchEncoding = Dict[str, torch.Tensor]


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


class SketchPredictor(nn.Module):
    def __init__(
        self,
        encoder_model: BertEncoder,
        hidden_size: int,
        token_embed_size: int,
        freeze_bert: bool = False,
        use_lstm_encoder: bool = False,
        dropout: float = 0.2
    ):
        nn.Module.__init__(self)

        self.encoder_model = encoder_model
        self.tokenizer = encoder_model.bert_model.tokenizer
        self.hidden_size = hidden_size
        self.freeze_bert = freeze_bert
        self.use_lstm_encoder = use_lstm_encoder

        downstream_encoder_output_size = encoder_model.output_size  # encoder_model.config.hidden_size

        if use_lstm_encoder:
            self.lstm_encoder = nn.LSTM(
                downstream_encoder_output_size, hidden_size,
                bidirectional=True, batch_first=True
            )

            self.src_encoding_size = hidden_size * 2
        else:
            self.src_encoding_size = downstream_encoder_output_size

        self.src_attention_value_to_key = nn.Linear(
            self.src_encoding_size,
            hidden_size, bias=False
        )

        self.column_encoding_with_feature = nn.Linear(
            self.src_encoding_size + self.encoder_model.column_feature_num,
            self.src_encoding_size,
            bias=False
        )

        self.column_attention_value_to_key = nn.Linear(
            self.src_encoding_size,
            hidden_size, bias=False
        )

        self.decoder_init_linear = nn.Linear(
            self.encoder_model.bert_model.config.hidden_size,
            self.hidden_size
        )

        self.decoder_lstm = nn.LSTMCell(
            hidden_size + token_embed_size,
            hidden_size
        )

        self.h_and_ctx_q_linear = nn.Linear(
            hidden_size + self.src_encoding_size, self.src_encoding_size,
            bias=False
        )

        self.decoder_att_vec_linear = nn.Linear(
            hidden_size + self.src_encoding_size * 2,
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
            initializer_range = self.encoder_model.bert_model.config.initializer_range

            if isinstance(_module, (nn.Linear, nn.Embedding)):
                _module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(_module, nn.Linear) and _module.bias is not None:
                _module.bias.data.zero_()

        for module_name, module in self.named_modules():
            if 'encoder_model' not in module_name:
                # print(module_name)
                module.apply(_init_weights)

        torch.nn.init.eye_(self.column_encoding_with_feature.weight)

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
            if params['sketch_decoder_use_parser_table_bert']:
                bert_model = kwargs['encoder'].bert_model
            else:
                tb_path = params.get('table_bert_model')
                if tb_path:
                    tb_path = Path(tb_path)
                    bert_model = TableBERT.load(
                        tb_path, tb_path.parent / 'tb_config.json',
                        column_representation=params['column_representation']
                    )
                else:
                    bert_model = TableBERT.from_pretrained(
                        params['bert_model'],
                        tokenizer=BertTokenizer.from_pretrained(params['bert_model']),
                        table_bert_config=json.load(open(params['table_bert_config_file'])),
                        column_representation=params['column_representation']
                    )
        else:
            bert_model = BertModel.from_pretrained(params['bert_model'])

        encoder_config = dict(config)
        downstream_encoder = BertEncoder(
            bert_model,
            output_size=params['sketch_decoder_hidden_size'],
            question_feat_size=config['n_en_input_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            column_feature_num=config['n_de_output_features'],
            dropout=config['dropout'],
            config=encoder_config
        )

        model = cls(
            downstream_encoder,
            hidden_size=params['sketch_decoder_hidden_size'],
            token_embed_size=params['sketch_decoder_token_embed_size'],
            freeze_bert=params['sketch_decoder_freeze_bert'],
            use_lstm_encoder=params['sketch_decoder_use_lstm_encoder']
        )

        return model

    @classmethod
    def default_params(cls):
        return {
            'sketch_decoder_use_table_bert': True,
            'sketch_decoder_use_parser_table_bert': True,
            'bert_model': 'bert-base-uncased',
            'sketch_decoder_hidden_size': 256,
            'sketch_decoder_token_embed_size': 128,
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

        # two step attention, first we attend to the question tokens,
        # then we attend to the columns and other memory values
        # (batch_size, question_len, encoding_size)
        ctx_q_t, alpha_q_t = nn_util.dot_prod_attention(
            query=h_t,
            keys=src_encodings['key'],
            values=src_encodings['value'],
            entry_masks=src_encodings['mask']
        )

        # # (batch_size, hidden_size)
        # att_q_t = torch.tanh(self.h_and_ctx_q_linear(
        #     torch.cat([h_t, ctx_q_t], dim=-1)))
        #
        # # (batch_size, column_num, encoding_size)
        # ctx_column_t, alpha_column_t = nn_util.dot_prod_attention(
        #     query=att_q_t,
        #     keys=column_encodings['key'],
        #     values=column_encodings['value'],
        #     entry_masks=column_encodings['mask']
        # )

        # (batch_size, column_num, encoding_size)
        ctx_column_t, alpha_column_t = nn_util.dot_prod_attention(
            query=h_t,
            keys=column_encodings['key'],
            values=column_encodings['value'],
            entry_masks=column_encodings['mask']
        )

        # # (batch_size, context_vector_size)
        ctx_t = torch.cat([ctx_q_t, ctx_column_t], dim=-1)

        # att_t = torch.tanh(self.decoder_att_vec_linear(torch.cat([h_t, ctx_column_t + att_q_t], 1)))
        att_t = torch.tanh(self.decoder_att_vec_linear(torch.cat([h_t, ctx_t], 1)))
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t

    def to_tensor_dict(self, env_contexts: List[Dict], sketches: List[Sketch] = None):
        batch_size = len(env_contexts)
        tensor_dict = {}

        if sketches is not None:
            max_sketch_len = max(2 + len(sketch.operators) for sketch in sketches)
            sketch_token_ids = np.zeros((batch_size, max_sketch_len), dtype=np.int64)
            sketch_mask = np.zeros((batch_size, max_sketch_len), dtype=np.float32)

            for i, (example, sketch) in enumerate(
                    zip(env_contexts, sketches)):

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

    def forward(self, env_contexts: List[Dict], sketches: List[Sketch], context_encoding: Dict = None):
        prediction_target = self.to_tensor_dict(
            env_contexts, sketches)

        # (batch_size, sequence_len, encoding_size)
        # (batch_size, max_column_len, encoding_size)
        src_encodings, column_encodings, decoder_init_state = self.encode(env_contexts, context_encoding)

        tgt_sketch_token_ids = prediction_target['tgt_sketch_token_ids']

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
        ).squeeze(-1) * prediction_target['tgt_mask'][:, 1:]

        sketch_prob = tgt_sketch_token_prob.sum(dim=-1)

        return sketch_prob

    def get_trajectory_sketch_prob(self, trajectories: List[Trajectory], context_encoding: Dict = None):
        # questions = [
        #     traj.context['question_tokens']
        #     for traj in trajectories
        # ]
        env_contexts = [
            traj.context
            for traj in trajectories
        ]

        sketches = [
            Sketch(traj.program)
            for traj in trajectories
        ]

        sketch_log_prob = self.forward(env_contexts, sketches, context_encoding)
        # sketch_loss = -sketch_log_prob.mean()

        return sketch_log_prob

    def encode(self, env_contexts: List, context_encoding: Dict = None):
        if context_encoding is None:
            context_encoding = self.encoder_model.encode(env_contexts)

        src_encodings = {
            'value': context_encoding['question_encoding'],
            'key': context_encoding['question_encoding_att_linear'],
            'mask': context_encoding['question_mask']
        }

        column_encoding_var = context_encoding['constant_encoding']

        # add output features here!
        output_features = np.zeros((
            len(env_contexts),
            self.encoder_model.memory_size - self.encoder_model.builtin_func_num,
            self.encoder_model.column_feature_num),
            dtype=np.float32
        )

        for i, env_context in enumerate(env_contexts):
            mem_id_feat_dict = env_context['id_feature_dict']
            # feat_num = len(next(mem_id_feat_dict.values()))
            constant_val_id_start = self.encoder_model.builtin_func_num
            # constant_val_num = len(mem_id_feat_dict) - self.encoder_model.builtin_func_num

            output_features[i, :] = [
                mem_id_feat_dict[mem_idx]
                for mem_idx
                in range(constant_val_id_start, self.encoder_model.memory_size)
            ]
        output_features = torch.from_numpy(output_features).to(self.device)
        column_encoding_var = torch.tanh(self.column_encoding_with_feature(
            torch.cat([column_encoding_var, output_features], dim=-1)))

        column_encodings = {
            'value': column_encoding_var,
            'key': self.column_attention_value_to_key(
                column_encoding_var),
            'mask': context_encoding['constant_mask']
        }

        # if self.freeze_bert:
        #     context_encoding['question_encoding'] = context_encoding['question_encoding'].detach()
        #     context_encoding['column_encoding'] = context_encoding['column_encoding'].detach()

        decoder_state_init_vec = context_encoding['cls_encoding']  # encoding of the [CLS] label
        decoder_init_state = self.decoder_init(decoder_state_init_vec)

        return src_encodings, column_encodings, decoder_init_state

    def _encode(self, input_ids, segment_ids, attention_mask):
        bert_sequence_output, _ = self.encoder_model(
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        # question_encoding, column_encoding, info = self.bert_model.encode([
        #     Example(question=e.context['question_tokens'], table=e.context['table'])
        #     for e in envs
        # ])
        #
        # context_encoding = {
        #     'question_encoding': question_encoding,
        #     'column_encoding': column_encoding,
        # }
        # context_encoding.update(info['tensor_dict'])

        env_context = [env.context for env in envs]
        src_encodings, column_encodings, decoder_init_state = self.encode(env_context)

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


class SketchEncoder(nn.Module):
    def __init__(
            self,
            output_size: int,
            embedding: nn.Embedding = None,
            embedding_size: int = None,
            vocab: Dict = None
    ):
        super(SketchEncoder, self).__init__()

        self.vocab = vocab
        self.embedding = embedding

        if self.vocab is None:
            self.executor_api = get_executor_api()
            operators = sorted(self.executor_api['func_dict'])
            self.vocab = {
                token: idx
                for idx, token
                in enumerate(['<s>', '</s>', 'v', '(', ')', '<END>'] + operators)
            }

        if embedding is None:
            self.embedding = nn.Embedding(len(self.vocab), embedding_size)

        self.output_size = output_size
        self.lstm = nn.LSTM(
            self.embedding.embedding_dim, output_size // 2, bidirectional=True, batch_first=True)

        self.init_weights()

    def init_weights(self):
        print('Init sketch encoder weights')

        def _init_weights(_module):
            if isinstance(_module, (nn.Linear, nn.Embedding)):
                _module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(_module, nn.Linear) and _module.bias is not None:
                _module.bias.data.zero_()

        for module in [
            self.embedding
        ]:
            module.apply(_init_weights)

    @classmethod
    def build(cls, config: Dict, sketch_predictor: SketchPredictor):
        output_size = config.get('en_embedding_size', 100)
        embedding_size = config.get('sketch_decoder_token_embed_size', 128)

        return cls(
            output_size,
            embedding_size=embedding_size
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to_input_tensor(self, sketches: List[Sketch]):
        batch_size = len(sketches)
        max_len = max(len(sketch) for sketch in sketches)

        token_id = np.zeros((batch_size, max_len), np.int64)
        token_mask = np.zeros((batch_size, max_len), np.float32)
        variable_mask = np.zeros((batch_size, max_len), np.float32)

        for i in range(batch_size):
            sketch = sketches[i]
            tokens = sketch.tokens
            for t, token in enumerate(tokens):
                is_slot = sketch.is_variable_slot(token)
                token_id[i, t] = self.vocab[token]
                variable_mask[i, t] = is_slot
            token_mask[i, :len(tokens)] = 1.

        token_id = torch.from_numpy(token_id).to(self.device)
        token_mask = torch.from_numpy(token_mask).to(self.device)
        variable_mask = torch.from_numpy(variable_mask).to(self.device)

        return token_id, token_mask, variable_mask

    def forward(self, sketches: List[Sketch]):
        sketch_lens = [len(sketch) for sketch in sketches]
        # (batch_size, max_sketch_len)
        token_id, token_mask, var_time_step_mask = self.to_input_tensor(sketches)

        # (batch_size, max_sketch_len, embedding_dim)
        token_embedding = self.embedding(token_id)

        packed_embedding = pack_padded_sequence(
            token_embedding, sketch_lens,
            batch_first=True, enforce_sorted=False
        )

        # (batch_size, max_sketch_len, output_size)
        token_encoding, (sorted_last_state, sorted_last_cell) = self.lstm(packed_embedding)
        token_encoding, _ = pad_packed_sequence(token_encoding, batch_first=True)
        token_encoding = token_encoding * token_mask.unsqueeze(-1)

        # (num_directions, batch_size, hidden_size)
        last_state = sorted_last_state.index_select(dim=1, index=packed_embedding.unsorted_indices)
        cat_last_state = torch.cat([last_state[0], last_state[1]], dim=-1)

        encoding_dict = {
            'value': token_encoding,
            'mask': token_mask,
            'var_time_step_mask': var_time_step_mask,
            'last_state': cat_last_state
        }

        return encoding_dict


class SketchManagerTrainer(object):
    def __init__(self, model: SketchPredictor, num_train_step: int, freeze_bert_for_niter: int, config: Dict):
        self.model = model

        no_grad = ['pooler']
        if self.model.use_table_bert:
            bert_params = no_decay = []
        else:
            bert_params = list([
                (p_name, p)
                for (p_name, p) in model.encoder_model.named_parameters()
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
