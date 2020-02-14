import multiprocessing
import sys
import time
from collections import namedtuple
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.utils
import numpy as np

from pytorch_pretrained_bert.modeling import BertModel
from typing import List, Tuple, Any, Dict, Optional

from nsm import nn_util
from nsm.env_factory import Environment, Trajectory
from nsm.execution.worlds.wikitablequestions import world_config
from nsm.parser_module.bert_encoder import BertEncoder
from nsm.parser_module.table_bert_helper import get_table_bert_model
from nsm.sketch.sketch import Sketch

SketchEncoding = Dict[str, torch.Tensor]


Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class SketchPredictor(nn.Module):
    def __init__(
        self,
        encoder_model: BertEncoder,
        hidden_size: int,
        token_embed_size: int,
        freeze_bert: bool = False,
        use_lstm_encoder: bool = False,
        dropout: float = 0.2,
        use_canonical_column_representation: bool = False,
        use_column_feature: bool = False
    ):
        nn.Module.__init__(self)

        self.encoder_model = encoder_model
        self.tokenizer = encoder_model.bert_model.tokenizer
        self.hidden_size = hidden_size
        self.freeze_bert = freeze_bert
        self.use_lstm_encoder = use_lstm_encoder
        self.use_canonical_column_representation = use_canonical_column_representation
        self.use_column_feature = use_column_feature

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

        col_feat_num = self.encoder_model.column_feature_num if self.use_column_feature else 0

        if use_canonical_column_representation:
            column_proj_in_feat = self.encoder_model.bert_model.bert_config.hidden_size + col_feat_num
        else:
            column_proj_in_feat = self.src_encoding_size + col_feat_num

        self.column_encoding_projection = nn.Linear(
            column_proj_in_feat,
            self.src_encoding_size,
            bias=False
        )

        self.column_attention_value_to_key = nn.Linear(
            self.src_encoding_size,
            hidden_size, bias=False
        )

        self.decoder_init_linear = nn.Linear(
            self.encoder_model.bert_model.bert_config.hidden_size,
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

        self.executor_api = world_config['executor_api']
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
            initializer_range = self.encoder_model.bert_model.bert_config.initializer_range

            if isinstance(_module, (nn.Linear, nn.Embedding)):
                _module.weight.data.normal_(mean=0.0, std=initializer_range)
            if isinstance(_module, nn.Linear) and _module.bias is not None:
                _module.bias.data.zero_()

        for module_name, module in self.named_modules():
            if 'bert_model' not in module_name:
                module.apply(_init_weights)

    @property
    def device(self):
        return next(self.parameters()).device

    @classmethod
    def build(cls, config, **kwargs):
        from nsm.execution.worlds.wikitablequestions import world_config as wikitablequestions_config
        config['builtin_func_num'] = wikitablequestions_config['interpreter_builtin_func_num']

        params = cls.default_params()
        params.update(config)

        if params['sketch_decoder_use_table_bert']:
            if params['sketch_decoder_use_parser_table_bert']:
                raise RuntimeError('Does not support this option!')
                # bert_model = kwargs['encoder'].bert_model
            else:
                bert_model = get_table_bert_model(
                    config,
                    use_proxy=False,
                    master='sketch_predictor'
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
            use_lstm_encoder=params['sketch_decoder_use_lstm_encoder'],
            use_canonical_column_representation=params['sketch_decoder_use_canonical_column_representation'],
            use_column_feature=params['sketch_decoder_use_column_feature']
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
            'sketch_decoder_use_lstm_encoder': False,
            'sketch_decoder_use_canonical_column_representation': False,
            'sketch_decoder_use_column_feature': False
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

    def forward(self, env_contexts: List[Dict], sketches: List[Sketch]):
        prediction_target = self.to_tensor_dict(
            env_contexts, sketches)

        # (batch_size, sequence_len, encoding_size)
        # (batch_size, max_column_len, encoding_size)
        src_encodings, column_encodings, decoder_init_state = self.encode(env_contexts)

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
        # print(self.device)
        # if self.device.index == 1:
        #     from fairseq import pdb
        #     pdb.set_trace()

        if context_encoding is None:
            context_encoding = self.encoder_model.encode(env_contexts)

        # for key, val in context_encoding.items():
        #     if torch.is_tensor(val):
        #         print(f'[SketchPredictor] {key}.device={val.device}', file=sys.stderr)
        #
        # print(f'[SketchPredictor] device={self.device}', file=sys.stderr)

        src_encodings = {
            'value': context_encoding['question_encoding'],
            'key': context_encoding['question_encoding_att_linear'],
            'mask': context_encoding['question_mask']
        }

        max_column_num = context_encoding['canonical_column_encoding'].size(1) \
            if self.use_canonical_column_representation \
            else context_encoding['column_encoding'].size(1)

        if self.use_column_feature:
            # add output features here!
            output_features = np.zeros(
                (
                    len(env_contexts),
                    max_column_num,
                    self.encoder_model.column_feature_num
                ),
                dtype=np.float32
            )

            for example_id, env_context in enumerate(env_contexts):
                table = env_context['table']
                column_info = env_context['table'].column_info
                mem_id_feat_dict = env_context['id_feature_dict']
                constant_val_id_start = self.encoder_model.builtin_func_num

                if self.use_canonical_column_representation:
                    canonical_column_num = len(table.header)
                    for col_idx in range(canonical_column_num):
                        raw_col_indices = [
                            idx
                            for idx, _col_idx in enumerate(column_info['raw_column_canonical_ids'])
                            if _col_idx == col_idx
                        ]
                        raw_col_features = np.array([
                            mem_id_feat_dict[constant_val_id_start + idx]
                            for idx in raw_col_indices]
                        )
                        col_features = raw_col_features.max(axis=0)  # (feature_num)

                        if col_idx < max_column_num:
                            output_features[example_id, col_idx] = col_features
                else:
                    output_features[example_id, :] = [
                        mem_id_feat_dict[mem_idx]
                        for mem_idx
                        in range(constant_val_id_start, constant_val_id_start + max_column_num)
                    ]

            output_features = torch.from_numpy(output_features).to(self.device)

            if self.use_canonical_column_representation:
                column_encoding_var = self.column_encoding_projection(torch.cat([
                    context_encoding['canonical_column_encoding'], output_features], dim=-1))
                column_mask = context_encoding['canonical_column_mask']
            else:
                column_encoding_var = self.column_encoding_projection(torch.cat([
                    context_encoding['column_encoding'], output_features], dim=-1))
                column_mask = context_encoding['column_mask']
        else:
            if self.use_canonical_column_representation:
                column_encoding_var = self.column_encoding_projection(
                    context_encoding['canonical_column_encoding'])
                column_mask = context_encoding['canonical_column_mask']
            else:
                column_encoding_var = self.column_encoding_projection(
                    context_encoding['column_encoding'])
                column_mask = context_encoding['column_mask']

        column_encodings = {
            'value': column_encoding_var,
            'key': self.column_attention_value_to_key(
                column_encoding_var),
            'mask': column_mask
        }

        # if self.freeze_bert:
        #     context_encoding['question_encoding'] = context_encoding['question_encoding'].detach()
        #     context_encoding['column_encoding'] = context_encoding['column_encoding'].detach()

        decoder_state_init_vec = context_encoding['cls_encoding']  # encoding of the [CLS] label
        decoder_init_state = self.decoder_init(decoder_state_init_vec)

        return src_encodings, column_encodings, decoder_init_state

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

    def get_sketches(self, env_context: List[Dict], K=5):
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

        src_encodings, column_encodings, decoder_init_state = self.encode(env_context)

        return self.beam_search(env_context, src_encodings, column_encodings, decoder_init_state, beam_size=K)

    def beam_search(
            self,
            examples: List[Dict],
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


class SketchPredictorProxy(object):
    def __init__(self):
        pass

    def initialize(self, actor):
        self.request_queue = actor.sketch_predictor_request_queue
        self.result_queue = actor.sketch_predictor_result_queue
        self.actor = actor

        self.worker_id = actor.actor_id

    def get_sketches(self, envs: List[Environment], K=5):
        env_context = [env.context for env in envs]

        request = {
            'worker_id': self.worker_id,
            'payload': (env_context, K)
        }

        self.request_queue.put(request)

        response = self.result_queue.get()
        sketches = response
        # sketches = response['sketches']

        return sketches


class SketchPredictorServer(multiprocessing.Process):
    def __init__(
        self,
        config: Dict,
        device: str
    ):
        super(SketchPredictorServer, self).__init__(daemon=True)
        
        self.request_queue = multiprocessing.Queue()
        self.workers = dict()

        self.config = config
        self.target_device = device

        self.model_path: Optional[str] = None
        self.learner_msg_val: multiprocessing.Value = None

    @property
    def device(self):
        return next(self.sketch_predictor.parameters()).device

    def register_worker(self, actor):
        sketch_predictor_result_queue = getattr(actor, 'sketch_predictor_result_queue', None)
        if not sketch_predictor_result_queue:
            sketch_predictor_result_queue = multiprocessing.Queue()
            setattr(actor, 'sketch_predictor_result_queue', sketch_predictor_result_queue)

        self.workers[actor.actor_id] = SimpleNamespace(
            result_queue=sketch_predictor_result_queue
        )
        setattr(actor, 'sketch_predictor_request_queue', self.request_queue)

    def init_server(self):
        target_device = self.target_device
        if 'cuda' in str(target_device):
            torch.cuda.set_device(target_device)

        self.sketch_predictor = SketchPredictor.build(
            self.config,
        ).to(target_device).eval()

    def run(self):
        print('[SketchPredictorServer] Init sketch predictor...', file=sys.stderr)
        self.init_server()
        print('[SketchPredictorServer] Init success', file=sys.stderr)

        cum_request_num = 0.
        cum_model_ver_not_match_num = 0.
        cum_process_time = 0.
        with torch.no_grad():
            while True:
                request = self.request_queue.get()

                cum_request_num += 1.

                payload = request['payload']
                worker_id = request['worker_id']

                t1 = time.time()
                encode_result = self.sketch_predictor.get_sketches(*payload)
                packed_result = self.pack_encode_result(encode_result)
                t2 = time.time()

                self.workers[worker_id].result_queue.put(packed_result)

                cum_process_time += t2 - t1
                if cum_request_num % 100 == 0:
                    print(f'[SketchPredictorServer] cum. request={cum_request_num}, '
                          f'speed={cum_request_num / cum_process_time} requests/s',
                          file=sys.stderr)

                self.check_and_load_new_model()

    def pack_encode_result(self, encode_result: Any) -> Any:
        def _to_numpy_array(obj):
            if isinstance(obj, tuple):
                return tuple(_to_numpy_array(x) for x in obj)
            elif isinstance(obj, list):
                return list(_to_numpy_array(x) for x in obj)
            elif isinstance(obj, dict):
                return {
                    key: _to_numpy_array(val)
                    for key, val
                    in obj.items()
                }
            elif torch.is_tensor(obj):
                return obj.cpu().numpy()
            else:
                return obj

        packed_result = _to_numpy_array(encode_result)

        return packed_result

    def check_and_load_new_model(self):
        new_model_path = self.learner_msg_val.value.decode()

        if new_model_path and new_model_path != self.model_path:
            t1 = time.time()

            state_dict = torch.load(new_model_path, map_location=lambda storage, loc: storage)

            self.sketch_predictor.load_state_dict(state_dict)
            self.model_path = new_model_path
            self.sketch_predictor.eval()

            t2 = time.time()
            print('[SketchPredictorServer] loaded new model [%s] (took %.2f s)' % (new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False
