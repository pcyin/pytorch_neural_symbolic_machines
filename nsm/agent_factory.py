"""Implementation of RL agents."""
import collections
import heapq
import itertools
import math
from collections import OrderedDict, namedtuple
import json
import sys
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import RNNCellBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsm import nn_util, data_utils
from nsm.embedding import EmbeddingModel, Embedder
from nsm.env_factory import Observation, Trajectory, Environment, QAProgrammingEnv

# Sample = collections.namedtuple('Sample', ['trajectory', 'prob'])


class Sample(object):
    def __init__(self, trajectory: Trajectory, prob: Union[float, torch.Tensor]):
        self.trajectory = trajectory
        self.prob = prob

    def to(self, device: torch.device):
        for ob in self.trajectory.observations:
            ob.to(device)

        return self

    def __repr__(self):
        return 'Sample({}, prob={})'.format(self.trajectory, self.prob)

    __str__ = __repr__


Hypothesis = collections.namedtuple('Hypothesis', ['env', 'score'])


class DecoderState(object):
    def __init__(self, state, memory):
        self.state = state
        self.memory = memory

    def __getitem__(self, indices):
        sliced_state = [(s[0][indices], s[1][indices]) for s in self.state]
        sliced_memory = self.memory[indices]

        return DecoderState(sliced_state, sliced_memory)


class Encoder(nn.Module):
    def __init__(self, question_feat_size, hidden_size, num_layers,
                 max_variable_num_on_memory,
                 context_embedder,
                 output_proj_size,
                 dropout=0.):
        super(Encoder, self).__init__()

        self.context_embedder = context_embedder

        self.lstm_encoder = nn.LSTM(input_size=context_embedder.embed_size + question_feat_size,
                                    hidden_size=hidden_size, num_layers=num_layers,
                                    batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        if output_proj_size > 0:
            self.output_proj_linear = nn.Linear(hidden_size * 2, hidden_size, bias=True)
            self.question_encoding_att_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        else:
            self.question_encoding_att_linear = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.max_variable_num_on_memory = max_variable_num_on_memory

    @property
    def device(self):
        return self.context_embedder.device

    def _context_list_to_batch_dict(self, env_contexts: List[Dict]) -> Dict:
        batch_dict = dict()
        keys = list(env_contexts[0].keys())
        for key in keys:
            val_list = [x[key] for x in env_contexts]
            if key in ('question_word_ids', ):
                batch_val_tensor = nn_util.to_padded_tensor(val_list, pad_id=0)
            elif key in ('constant_value_embeddings', 'constant_spans', 'question_features'):
                # (batch_size, max_entry_num, entry_dim)
                if key == 'question_features':
                    max_entry_num = max(len(val) for val in val_list)
                else:
                    max_entry_num = self.max_variable_num_on_memory

                entry_dim = len(val_list[0][0])
                batch_size = len(env_contexts)
                batch_val_tensor = np.zeros((batch_size, max_entry_num, entry_dim),
                                            dtype=np.int64 if key == 'constant_spans' else np.float32)
                if key == 'constant_spans':
                    batch_val_tensor.fill(-1.)

                for i, val in enumerate(val_list):
                    entry_num = len(val)
                    batch_val_tensor[i, :entry_num] = val

                batch_val_tensor = torch.from_numpy(batch_val_tensor)
            else:
                continue

            batch_dict[key] = batch_val_tensor.to(self.device)

        # get question mask
        question_lengths = [len(x['question_tokens']) for x in env_contexts]
        batch_dict['question_lengths'] = question_lengths
        question_mask = nn_util.get_sequence_mask_from_length_array(question_lengths).to(self.device)
        batch_dict['question_mask'] = question_mask

        return batch_dict

    @staticmethod
    def build(config, params=None):
        if params is None:
            # Load pretrained embeddings.
            embedding_model = EmbeddingModel(config['vocab_file'], config['embedding_file'])

            # create vocabulary
            vocab = json.load(open(config['en_vocab_file']))
            en_vocab = data_utils.Vocab([])
            en_vocab.load_vocab(vocab)
            print('{} unique tokens in encoder vocab'.format(len(en_vocab.vocab)))
            trainable_token_num = len(en_vocab.special_tks)

            from table import utils
            pretrained_embeddings = []
            for i in range(len(en_vocab.special_tks), en_vocab.size):
                pretrained_embeddings.append(
                    utils.average_token_embedding(
                        utils.find_tk_in_model(
                            en_vocab.lookup(i, reverse=True), embedding_model),
                        embedding_model,
                        embedding_size=config['pretrained_embedding_size']))
            pretrained_embeddings = torch.tensor(pretrained_embeddings)
        else:
            pretrained_embeddings = params['encoder.context_embedder.pretrained_embedding.weight']
            trainable_token_num = params['encoder.context_embedder.trainable_embedding.weight'].size(0)

        embedder = Embedder(trainable_token_num=trainable_token_num,
                            embed_size=config['en_embedding_size'],
                            pretrained_embedding=pretrained_embeddings)

        encoder = Encoder(question_feat_size=config['n_en_input_features'],
                          hidden_size=config['hidden_size'],
                          output_proj_size=config['hidden_size'],
                          num_layers=config['en_n_layers'],
                          context_embedder=embedder,
                          max_variable_num_on_memory=config['memory_size'] - config['builtin_func_num'],
                          dropout=config['dropout'])

        return encoder

    def encode_question(self, env_contexts: Dict):
        # (batch_size, question_len)
        questions = env_contexts['question_word_ids']

        # (batch_size, question_len, embed_size)
        question_embedding = self.context_embedder(questions)

        # (batch_size, question_len, question_feat_len)
        question_feat = env_contexts['question_features']

        # (batch_size, question_len, embed_size + question_feat_len)
        encoder_input = torch.cat([question_embedding, question_feat], dim=-1)

        # (batch_size, question_len)
        question_mask = env_contexts['question_mask']
        # (batch_size)
        question_lengths = env_contexts['question_lengths']

        sorted_seqs, sorted_seq_lens, restoration_indices, sorting_indices = nn_util.sort_batch_by_length(encoder_input,
                                                                                                          questions.new_tensor(question_lengths))

        packed_question_embedding = pack_padded_sequence(sorted_seqs, sorted_seq_lens.data.tolist(), batch_first=True)

        sorted_question_encodings, (last_states, last_cells) = self.lstm_encoder(packed_question_embedding)
        sorted_question_encodings, _ = pad_packed_sequence(sorted_question_encodings, batch_first=True)

        # apply dropout to the last layer
        # (batch_size, seq_len, hidden_size * 2)
        sorted_question_encodings = self.dropout(sorted_question_encodings)

        # (batch_size, question_len, hidden_size * 2)
        question_encodings = sorted_question_encodings.index_select(dim=0, index=restoration_indices)

        # (num_layers, direction_num, batch_size, hidden_size)
        last_states = last_states.view(self.lstm_encoder.num_layers, 2, -1, self.lstm_encoder.hidden_size)
        last_states = last_states.index_select(dim=2, index=restoration_indices)
        last_cells = last_cells.view(self.lstm_encoder.num_layers, 2, -1, self.lstm_encoder.hidden_size)
        last_cells = last_cells.index_select(dim=2, index=restoration_indices)

        # (batch_size, hidden_size)
        # concatenate forward and backward cell
        encoder_last_forward_states = []
        encoder_last_backward_states = []
        for i in range(self.lstm_encoder.num_layers):
            last_fwd_cell_i = last_cells[i, 0]
            last_bak_cell_i = last_cells[i, 1]
            # last_cell_i = torch.cat([last_fwd_cell_i, last_bak_cell_i], dim=-1)

            last_fwd_state_i = last_states[i, 0]
            last_bak_state_i = last_states[i, 1]
            # last_state_i = torch.cat([last_fwd_state_i, last_bak_state_i], dim=-1)

            # encoder_last_states.append((last_state_i, last_cell_i))
            encoder_last_forward_states.append((last_fwd_state_i, last_fwd_cell_i))
            encoder_last_backward_states.append((last_bak_state_i, last_bak_cell_i))

        encoder_last_states = (encoder_last_forward_states, encoder_last_backward_states)

        if hasattr(self, 'output_proj_linear'):
            question_encodings = self.output_proj_linear(question_encodings)

        return question_encodings, question_mask, encoder_last_states

    def encode(self, env_contexts: List[Dict]) -> Dict:
        context_encoding = self._context_list_to_batch_dict(env_contexts)
        question_encoding, question_mask, encoder_last_states = self.encode_question(context_encoding)

        # (batch_size, question_len, decoder_output_size)
        context_encoding['question_encoding'] = question_encoding
        # (batch_size, question_len)
        context_encoding['question_mask'] = question_mask
        # (batch_size, encoder_layers, 2, hidden_size)
        context_encoding['encoder_last_states'] = encoder_last_states

        # prepare for attention's linear transformation
        # (batch_size, question_len, decode_output_size)
        context_encoding['question_encoding_att_linear'] = self.question_encoding_att_linear(question_encoding)

        return context_encoding


class MultiLayerDropoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0., use_skip_connection=True):
        super(MultiLayerDropoutLSTMCell, self).__init__()

        cells = []
        for i in range(num_layers):
            x_dim = input_size if i == 0 else hidden_size
            cell = nn.LSTMCell(x_dim, hidden_size)
            cells.append(cell)

        self.num_layers = num_layers
        self.cell_list = nn.ModuleList(cells)
        self.dropout = nn.Dropout(dropout)
        self.use_skip_connection = use_skip_connection

    def forward(self, x, s_tm1):
        # x: (batch_size, input_size)
        o_i = None
        state = []
        for i in range(self.num_layers):
            h_i, c_i = self.cell_list[i](x, s_tm1[i])

            if i > 0 and self.use_skip_connection:
                o_i = h_i + x
            else:
                o_i = h_i

            o_i = self.dropout(o_i)

            s_i = (h_i, c_i)
            state.append(s_i)

            x = o_i

        return o_i, state


class Decoder(nn.Module):
    def __init__(self, mem_item_embed_size,
                 constant_value_embed_size,
                 encoder_hidden_size,
                 hidden_size,
                 num_layers, output_feature_num,
                 builtin_func_num,
                 memory_size,
                 att_type='dot_prod',
                 dropout=0.):
        super(Decoder, self).__init__()

        self.memory_size = memory_size

        self.decoder_cell_init_linear = nn.Linear(encoder_hidden_size * 2, hidden_size)

        self.rnn_cell = MultiLayerDropoutLSTMCell(mem_item_embed_size, hidden_size,
                                                  num_layers=num_layers, dropout=dropout)

        self.att_vec_linear = nn.Linear(encoder_hidden_size + hidden_size, hidden_size, bias=False)

        if att_type == 'dot_prod':
            self.attention_func = self.dot_prod_attention
        if att_type == 'bahdanau':
            self.att_l2_linear = nn.Linear(hidden_size, 1, bias=False)
            self.query_proj_linear = nn.Linear(hidden_size, hidden_size, bias=True)

            self.attention_func = self.bahdanau_attention

        self.constant_value_embedding_linear = nn.Linear(constant_value_embed_size, mem_item_embed_size)

        # (builtin_func_num, embed_size)
        self.builtin_func_embeddings = nn.Embedding(builtin_func_num, mem_item_embed_size)

        self.output_feature_linear = nn.Linear(output_feature_num, 1, bias=False)
        torch.nn.init.zeros_(self.output_feature_linear.weight)

        self.dropout = nn.Dropout(dropout)

    @property
    def device(self):
        return self.decoder_cell_init_linear.weight.device

    def get_initial_state(self, context_encoding):
        # prepare decoder's initial memory and internal LSTM state

        # (batch_size, mem_size, embed_size)
        constant_value_embedding = context_encoding['constant_value_embeddings']
        constant_value_embedding = self.constant_value_embedding_linear(constant_value_embedding)
        # (batch_size, mem_size, 2)
        constant_span = context_encoding['constant_spans']
        # (batch_size, mem_size)
        constant_span_mask = torch.ge(constant_span, 0)[:, :, 0].float()
        # mask out entries <= 0
        constant_span = constant_span * constant_span_mask.unsqueeze(-1).long()

        constant_span_size = constant_span.size()
        mem_size = constant_span_size[1]
        embed_size = constant_value_embedding.size(-1)

        question_encoding = context_encoding['question_encoding']
        batch_size = question_encoding.size(0)

        # (batch_size, mem_size, 2, embed_size)
        constant_span_embedding = torch.gather(question_encoding.unsqueeze(1).expand(-1, mem_size, -1, -1),
                                               index=constant_span.unsqueeze(-1).expand(-1, -1, -1, embed_size),
                                               dim=2)
        # (batch_size, mem_size, embed_size)
        constant_span_embedding = torch.mean(constant_span_embedding, dim=-2)
        constant_span_embedding = constant_span_embedding * constant_span_mask.unsqueeze(-1)

        constant_embedding = constant_value_embedding + constant_span_embedding

        # add built-in function embeddings
        # (batch_size, builtin_func_num, embed_size)
        builtin_func_embedding = self.builtin_func_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # (batch_size, builtin_func_num + mem_size, embed_size)
        initial_memory = torch.cat([builtin_func_embedding, constant_embedding], dim=1)[:, :self.memory_size]

        encoder_last_states = context_encoding['encoder_last_states']

        decoder_init_states = []
        for i in range(len(encoder_last_states[0])):
            (fwd_h_i, fwd_c_i), (bak_h_i, bak_c_i) = encoder_last_states[0][i], encoder_last_states[1][i]
            c_i = torch.cat([fwd_c_i, bak_c_i], dim=-1)
            sc_0_i = self.decoder_cell_init_linear(c_i)
            sh_0_i = torch.tanh(sc_0_i)
            decoder_init_states.append((sh_0_i, sc_0_i))

        # # using forward encoder state to initialize decoder
        # decoder_init_states = encoder_last_states[0]

        state = DecoderState(state=decoder_init_states, memory=initial_memory)

        return state

    def step(self, x: Union[List[Observation], Observation], state_tm1: DecoderState, context_encoding: Dict):
        """Perform one step of the decoder"""

        # first convert listed input to batched ones
        if isinstance(x, list):
            x = Observation.to_batched_input(x, memory_size=self.memory_size)

        batch_size = x.read_ind.size(0)

        # collect y_tm1 as inputs to inner rnn cells
        # Memory: (batch_size, mem_size, mem_value_dim)
        # (batch_size, mem_value_dim)
        input_mem_entry = state_tm1.memory[torch.arange(batch_size), x.read_ind]

        # (batch_size, hidden_size)
        inner_output_t, inner_state_t = self.rnn_cell(input_mem_entry, state_tm1.state)

        # attention over context
        ctx_t, alpha_t = self.attention_func(query=inner_output_t,
                                             keys=context_encoding['question_encoding_att_linear'],
                                             values=context_encoding['question_encoding'],
                                             entry_masks=context_encoding['question_mask'])

        # (batch_size, hidden_size)
        att_t = torch.tanh(self.att_vec_linear(torch.cat([inner_output_t, ctx_t], 1)))  # E.q. (5)
        # att_t = self.dropout(att_t)

        # compute scores over valid memory entries
        # memory is organized by:
        # [built-in functions, constants and variables]

        # dot product attention
        # (batch_size, mem_size)
        mem_logits = torch.matmul(state_tm1.memory, att_t.unsqueeze(-1)).squeeze(-1)

        # add output features to logits
        # (batch_size, mem_size)
        output_feature = self.output_feature_linear(x.output_features).squeeze(-1)

        mem_logits = mem_logits + output_feature

        # write head of shape (batch_size)
        # mask of shape (batch_size)
        write_mask = torch.ge(x.write_ind, 0).float()
        # mask out negative entries in write_ind
        write_ind = x.write_ind * write_mask.long()
        # (batch_size, hidden_size)
        write_value = att_t * write_mask.unsqueeze(-1)

        # write to memory
        memory_tm1 = state_tm1.memory
        memory_t = memory_tm1.scatter_add(1, write_ind.view(-1, 1, 1).expand(-1, -1, memory_tm1.size(-1)), write_value.unsqueeze(1))

        state_t = DecoderState(state=inner_state_t, memory=memory_t)

        return mem_logits, state_t

    def step_and_get_action_scores_t(self, observations_t, state_tm1, context_encoding):
        mem_logits, state_t = self.step(observations_t, state_tm1, context_encoding=context_encoding)

        # (batch_size, mem_size)
        action_score_t = nn_util.masked_log_softmax(mem_logits, mask=observations_t.valid_action_mask)

        return action_score_t, state_t

    def bahdanau_attention(self, query, keys, values, entry_masks):
        query = self.query_proj_linear(query)
        # (batch_size, src_sent_len)
        att_weight = self.att_l2_linear(torch.tanh(query.unsqueeze(1) + keys)).squeeze(-1)

        if entry_masks is not None:
            att_weight.data.masked_fill_((1.0 - entry_masks).byte(), -float('inf'))

        att_prob = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_prob.view(*att_view), values).squeeze(1)

        return ctx_vec, att_prob

    def dot_prod_attention(self, query: torch.Tensor,
                           keys: torch.Tensor,
                           values: torch.Tensor,
                           entry_masks: torch.Tensor=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)

        if entry_masks is not None:
            att_weight.data.masked_fill_((1.0 - entry_masks).byte(), -float('inf'))

        att_prob = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_prob.view(*att_view), values).squeeze(1)

        return ctx_vec, att_prob

    @staticmethod
    def build(config):
        return Decoder(mem_item_embed_size=config['value_embedding_size'],
                       constant_value_embed_size=config['pretrained_embedding_size'],
                       encoder_hidden_size=config['hidden_size'],
                       hidden_size=config['hidden_size'],
                       num_layers=config['n_layers'],
                       output_feature_num=config['n_de_output_features'],
                       builtin_func_num=config['builtin_func_num'],
                       memory_size=config['memory_size'],
                       att_type=config['attention_type'],
                       dropout=config['dropout'])


class PGAgent(nn.Module):
    "Agent trained by policy gradient."

    def __init__(self, encoder: Encoder, decoder: Decoder, config: Dict, discount_factor: float = 1.0):
        super(PGAgent, self).__init__()

        self.config = config
        self.discount_factor = discount_factor

        self.encoder = encoder
        self.decoder = decoder

    @property
    def memory_size(self):
        return self.decoder.memory_size

    @property
    def device(self):
        return self.encoder.device

    def encode(self, env_context):
        return self.encoder.encode(env_context)

    def compute_trajectory_actions_prob(self, trajectories: List[Trajectory], return_info=False) -> torch.Tensor:
        contexts = [traj.context for traj in trajectories]
        context_encoding = self.encoder.encode(contexts)
        state_tm1 = init_state = self.decoder.get_initial_state(context_encoding)

        batched_observation_seq, tgt_actions_info = Trajectory.to_batched_sequence_tensors(trajectories, self.memory_size)

        # moved to device
        batched_observation_seq.to(self.device)
        # for val in tgt_actions_info.values(): val.to(self.device)
        # batched_observation_seq = Observation.to_batched_sequence_input(obs_seq, memory_size=self.memory_size)

        # tgt_action_id (batch_size, max_action_len)
        # tgt_action_mask (batch_size, max_action_len)
        tgt_action_id, tgt_action_mask = tgt_actions_info['tgt_action_ids'], tgt_actions_info['tgt_action_mask']
        tgt_action_id = tgt_action_id.to(self.device)
        tgt_action_mask = tgt_action_mask.to(self.device)

        max_time_step = batched_observation_seq.read_ind.size(1)
        action_logits = []
        for t in range(max_time_step):
            obs_slice_t = batched_observation_seq.slice(t)

            # mem_logits: (batch_size, memory_size)
            mem_logits, state_t = self.decoder.step(obs_slice_t, state_tm1, context_encoding)

            action_logits.append(mem_logits)
            state_tm1 = state_t

        # (max_action_len, batch_size, memory_size)
        action_logits = torch.stack(action_logits, dim=0).permute(1, 0, 2)

        # (batch_size, max_action_len, memory_size)
        action_log_probs = nn_util.masked_log_softmax(action_logits, batched_observation_seq.valid_action_mask)

        # (batch_size, max_action_len)
        tgt_action_log_probs = torch.gather(action_log_probs, dim=-1, index=tgt_action_id.unsqueeze(-1)).squeeze(-1) * tgt_action_mask

        if return_info:
            info = dict(
                action_log_probs=action_log_probs,
                tgt_action_id=tgt_action_id,
                tgt_action_mask=tgt_action_mask,
                action_logits=action_logits,
                valid_action_mask=batched_observation_seq.valid_action_mask
            )

            return tgt_action_log_probs, info

        return tgt_action_log_probs

    def compute_trajectory_prob(self, trajectories: List[Trajectory], log=True) -> torch.Tensor:
        with torch.no_grad():
            traj_log_prob = self.forward(trajectories)

            if not log:
                traj_log_prob = traj_log_prob.exp()

            return traj_log_prob.tolist()

    def forward(self, trajectories: List[Trajectory], entropy=False):
        # (batch_size, max_action_len)
        tgt_action_log_probs, meta_info = self.compute_trajectory_actions_prob(trajectories, return_info=True)

        # (batch_size)
        traj_log_prob = tgt_action_log_probs.sum(dim=-1)

        # compute entropy
        if entropy:
            # (batch_size, max_action_len, memory_size)
            logits = meta_info['action_logits']
            action_log_probs = meta_info['action_log_probs']
            # (batch_size, max_action_len, memory_size)
            valid_action_mask = meta_info['valid_action_mask']
            # (batch_size, max_action_len)
            tgt_action_mask = meta_info['tgt_action_mask']

            # masked_logits = logits * tgt_action_mask + (1. - tgt_action_mask) * -1.e30  # mask logits with a very negative number

            # max_z, pos = torch.max(masked_logits, dim=-1, keepdim=True)
            # z = masked_logits - max_z
            # exp_z = torch.exp(z)
            # (batch_size, max_action_len)
            # sum_exp_z = torch.sum(exp_z, dim=-1, keepdim=True)

            p_action = nn_util.masked_softmax(logits, mask=valid_action_mask)
            # neg_log_action = torch.log(sum_exp_z) - z

            H = - p_action * action_log_probs * valid_action_mask
            # H = p_action * neg_log_action
            H = torch.sum(H, dim=-1).sum(dim=-1) / tgt_action_mask.sum(-1)

            return traj_log_prob, H

        return traj_log_prob

    def sample_gpu(self, environments, sample_num, use_cache=False):
        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        duplicated_envs = []
        for env in environments:
            for i in range(sample_num):
                duplicated_envs.append(env.clone())

        environments = duplicated_envs
        for env in environments:
            env.use_cache = use_cache

        env_context = [env.get_context() for env in environments]
        context_encoding = self.encode(env_context)

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding)
        sample_probs = torch.zeros(len(environments), device=self.device)

        active_env_ids = set(range(len(environments)))
        while True:
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(self.device)
            mem_logits, state_t = self.decoder.step(observations_tm1, state_tm1, context_encoding=context_encoding)

            # (batch_size)
            sampled_action_t_id, sampled_action_t_prob = self.sample_action(mem_logits, batched_ob_tm1.valid_action_mask,
                                                                            return_log_prob=True)

            observations_t = []
            new_active_env_ids = set()
            for env_id, (env, action_t) in enumerate(zip(environments, sampled_action_t_id.tolist())):
                if env_id in active_env_ids:
                    action_rel_id = env.valid_actions.index(action_t)
                    ob_t, _, _, info = env.step(action_rel_id)
                    if env.done:
                        observations_t.append(observations_tm1[env_id])
                    else:
                        # if the ob_t.valid_action_indices is empty, then the environment will terminate automatically,
                        # so these is not need to check if this field is empty.
                        observations_t.append(ob_t)
                        new_active_env_ids.add(env_id)
                else:
                    observations_t.append(observations_tm1[env_id])

            sample_probs = sample_probs + sampled_action_t_prob
            # print(sample_probs)

            if new_active_env_ids:
                # context_encoding = nn_util.dict_index_select(context_encoding, active_env_ids)
                # observations_tm1 = [observations_t[i] for i in active_env_ids]
                # state_tm1 = state_t[active_env_ids]
                observations_tm1 = observations_t
                state_tm1 = state_t
                active_env_ids = new_active_env_ids
            else:
                break

        samples = []
        for env_id, env in enumerate(environments):
            if not env.error:
                traj = Trajectory.from_environment(env)
                samples.append(Sample(trajectory=traj, prob=sample_probs[env_id].item()))

        return samples

    def sample(self, environments, sample_num, use_cache=False):
        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        duplicated_envs = []
        for env in environments:
            for i in range(sample_num):
                duplicated_envs.append(env.clone())

        environments = duplicated_envs
        for env in environments:
            env.use_cache = use_cache

        completed_envs = []
        active_envs = environments

        env_context = [env.get_context() for env in environments]
        context_encoding = self.encode(env_context)

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding)
        sample_probs = torch.zeros(len(environments), device=self.device)

        while True:
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(self.device)
            mem_logits, state_t = self.decoder.step(observations_tm1, state_tm1, context_encoding=context_encoding)

            # try:
            # (batch_size)
            sampled_action_t_id, sampled_action_t_prob = self.sample_action(mem_logits, batched_ob_tm1.valid_action_mask,
                                                                            return_log_prob=True)
            # except RuntimeError:
            #     for ob in observations_tm1:
            #         print(f'Observation {ob}', file=sys.stderr)
            #         print(ob.valid_action_indices, file=sys.stderr)
            #
            #     print(batched_ob_tm1.valid_action_mask, file=sys.stderr)
            #     torch.save((mem_logits, batched_ob_tm1.valid_action_mask), 'tmp.bin')
            #     exit(-1)

            sample_probs = sample_probs + sampled_action_t_prob

            # print(sample_probs)

            observations_t = []
            new_active_env_pos = []
            new_active_envs = []
            has_completed_sample = False
            for env_id, (env, action_t) in enumerate(zip(active_envs, sampled_action_t_id.tolist())):
                action_rel_id = env.valid_actions.index(action_t)
                ob_t, _, _, info = env.step(action_rel_id)
                if env.done:
                    completed_envs.append((env, sample_probs[env_id].item()))
                    has_completed_sample = True
                else:
                    # if the ob_t.valid_action_indices is empty, then the environment will terminate automatically,
                    # so these is not need to check if this field is empty.
                    observations_t.append(ob_t)
                    new_active_env_pos.append(env_id)
                    new_active_envs.append(env)

            if not new_active_env_pos:
                break

            if has_completed_sample:
                # need to perform slicing
                context_encoding['question_encoding'] = context_encoding['question_encoding'][new_active_env_pos]
                context_encoding['question_mask'] = context_encoding['question_mask'][new_active_env_pos]
                context_encoding['question_encoding_att_linear'] = context_encoding['question_encoding_att_linear'][new_active_env_pos]

                state_tm1 = state_t[new_active_env_pos]
                sample_probs = sample_probs[new_active_env_pos]
            else:
                state_tm1 = state_t

            observations_tm1 = observations_t
            active_envs = new_active_envs

        samples = []
        for env_id, (env, prob) in enumerate(completed_envs):
            if not env.error:
                traj = Trajectory.from_environment(env)
                samples.append(Sample(trajectory=traj, prob=prob))

        return samples

    def new_beam_search(self, environments, beam_size, use_cache=False, return_list=False):
        # if already explored everything, then don't explore this environment anymore.
        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        CandidateHyp = collections.namedtuple('CandidateHyp',
                                              ['prev_hyp_env', 'rel_action_id', 'score', 'prev_hyp_abs_pos'])

        batch_size = len(environments)
        # max_live_hyp_num = 1
        # live_beam_names = [env.name for env in environments]

        beams = OrderedDict((env.name, [Hypothesis(env=env, score=0.)]) for env in environments)
        completed_hyps = OrderedDict((env.name, []) for env in environments)
        # empty_hyp = dict(env=None, score=float('-inf'), ob=Observation.empty(), parent_beam_abs_pos=0)

        # (env_num, ...)
        env_context = [env.get_context() for env in environments]
        context_encoding_expanded = context_encoding = self.encode(env_context)

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding)
        hyp_scores_tm1 = torch.zeros(batch_size, device=self.device)

        while beams:
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(self.device)

            # (hyp_num, memory_size)
            action_probs_t, state_t = self.decoder.step_and_get_action_scores_t(batched_ob_tm1, state_tm1,
                                                                                context_encoding=context_encoding_expanded)
            action_probs_t[(1 - batched_ob_tm1.valid_action_mask).byte()] = float('-inf')

            # (hyp_num, memory_size)
            cont_cand_hyp_scores = action_probs_t + hyp_scores_tm1.unsqueeze(-1)
            cont_cand_hyp_scores = cont_cand_hyp_scores

            # collect hypotheses
            beam_start = 0
            continuing_candidates = OrderedDict()
            new_beams = OrderedDict()

            observations_t = []
            new_hyp_parent_abs_pos_list = []
            new_hyp_scores = []
            for env_name, beam in beams.items():
                live_beam_size = len(beam)
                beam_end = beam_start + live_beam_size
                # (beam_size, memory_size)
                beam_new_cont_scores = cont_cand_hyp_scores[beam_start: beam_end]
                continuing_candidates[env_name] = []

                for prev_hyp_id, prev_hyp in enumerate(beam):
                    _cont_action_scores = beam_new_cont_scores[prev_hyp_id][prev_hyp.env.obs[-1].valid_action_indices].cpu()
                    for rel_action_id, new_hyp_score in enumerate(_cont_action_scores):
                        new_hyp_score = new_hyp_score.item()
                        if not math.isinf(new_hyp_score):
                            continuing_candidates[env_name].append(CandidateHyp(prev_hyp_env=prev_hyp.env,
                                                                                rel_action_id=rel_action_id,
                                                                                score=new_hyp_score,
                                                                                prev_hyp_abs_pos=beam_start + prev_hyp_id))

                # rank all hypotheses together with completed ones
                all_candidates = completed_hyps[env_name] + continuing_candidates[env_name]
                top_k_candidates = heapq.nlargest(beam_size, all_candidates, key=lambda x: x.score)
                completed_hyps[env_name] = []

                for cand_hyp in top_k_candidates:
                    if isinstance(cand_hyp, Hypothesis):
                        completed_hyps[env_name].append(cand_hyp)
                    else:
                        new_hyp_env = cand_hyp.prev_hyp_env.clone()

                        ob_t, _, _, info = new_hyp_env.step(cand_hyp.rel_action_id)

                        if new_hyp_env.done:
                            if not new_hyp_env.error:
                                new_hyp = Hypothesis(env=new_hyp_env, score=cand_hyp.score)
                                completed_hyps[new_hyp_env.name].append(new_hyp)
                        else:
                            new_hyp = Hypothesis(env=new_hyp_env, score=cand_hyp.score)
                            new_beams.setdefault(env_name, []).append(new_hyp)

                            new_hyp_parent_abs_pos_list.append(cand_hyp.prev_hyp_abs_pos)
                            observations_t.append(ob_t)
                            new_hyp_scores.append(cand_hyp.score)

                beam_start = beam_end

            if len(new_beams) == 0:
                break

            new_hyp_state_t = [(s[0][new_hyp_parent_abs_pos_list], s[1][new_hyp_parent_abs_pos_list]) for s in state_t.state]
            new_hyp_memory_t = state_t.memory[new_hyp_parent_abs_pos_list]

            state_tm1 = DecoderState(state=new_hyp_state_t, memory=new_hyp_memory_t)
            observations_tm1 = observations_t
            hyp_scores_tm1 = torch.tensor(new_hyp_scores, device=self.device)

            for key in context_encoding_expanded:
                if key in {'question_encoding', 'question_mask', 'question_encoding_att_linear'}:
                    tensor = context_encoding_expanded[key]
                    context_encoding_expanded[key] = tensor[new_hyp_parent_abs_pos_list]

            beams = new_beams

        if not return_list:
            # rank completed hypothesis
            for env_name in completed_hyps.keys():
                sorted_hyps = sorted(completed_hyps[env_name], key=lambda hyp: hyp.score, reverse=True)[:beam_size]
                completed_hyps[env_name] = [Sample(trajectory=Trajectory.from_environment(hyp.env), prob=hyp.score) for
                                            hyp in sorted_hyps]

            return completed_hyps
        else:
            samples_list = []
            for _hyps in completed_hyps.values():
                samples = [Sample(trajectory=Trajectory.from_environment(hyp.env), prob=hyp.score) for hyp in _hyps]
                samples_list.extend(samples)

            return samples_list

    def beam_search(self, environments, beam_size, use_cache=False):
        # if already explored everything, then don't explore this environment anymore.
        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        batch_size = len(environments)
        max_live_hyp_num = 1
        live_beam_names = [env.name for env in environments]

        beams = OrderedDict((env.name, [dict(env=env, score=0.)]) for env in environments)
        completed_hyps = OrderedDict((env.name, []) for env in environments)
        empty_hyp = dict(env=None, score=float('-inf'), ob=Observation.empty(), parent_beam_abs_pos=0)

        # (env_num, ...)
        env_context = [env.get_context() for env in environments]
        context_encoding_expanded = context_encoding = self.encode(env_context)

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding)
        hyp_scores_tm1 = torch.zeros(batch_size, device=self.device)

        def _expand_context(_ctx_encoding, _live_beam_ids, _max_live_hyp_num):
            _expand_ctx_dict = dict()

            for key, tensor in _ctx_encoding.items():
                if key in {'question_encoding', 'question_mask', 'question_encoding_att_linear'}:  # don't need this
                    if len(_live_beam_ids) < batch_size:
                        tensor = tensor[_live_beam_ids]

                    new_tensor_size = list(tensor.size())
                    new_tensor_size.insert(1, _max_live_hyp_num)
                    exp_tensor = tensor.unsqueeze(1).expand(*new_tensor_size).contiguous().view(*([-1] + new_tensor_size[2:]))

                    _expand_ctx_dict[key] = exp_tensor

            return _expand_ctx_dict

        while beams:
            live_beam_num = len(beams)
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(self.device)

            # (live_beam_num * max_live_hyp_num, memory_size)
            # (live_beam_num * max_live_hyp_num, ...)
            action_probs_t, state_t = self.decoder.step_and_get_action_scores_t(batched_ob_tm1, state_tm1,
                                                                                context_encoding=context_encoding_expanded)
            action_probs_t[(1 - batched_ob_tm1.valid_action_mask).byte()] = float('-inf')

            new_hyp_scores = action_probs_t + hyp_scores_tm1.unsqueeze(-1)
            # (live_beam_num, sorted_cand_list_size)
            sorted_cand_list_size = beam_size
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(new_hyp_scores.view(live_beam_num, -1), k=sorted_cand_list_size, dim=-1)   # have some buffer since not all valid actions will execute without error

            # (live_beam_num, sorted_cand_list_size)
            prev_hyp_ids = (top_cand_hyp_pos / self.memory_size).cpu()
            hyp_action_ids = (top_cand_hyp_pos % self.memory_size).cpu()
            top_cand_hyp_scores = top_cand_hyp_scores.cpu()  # move this tensor to cpu for fast indexing

            new_beams = OrderedDict()
            for beam_id, (env_name, beam) in enumerate(beams.items()):
                live_beam_size = beam_size - len(completed_hyps[env_name])
                for cand_idx in range(sorted_cand_list_size):
                    # if this is a valid action, create a new continuating hypothesis
                    # otherwise, the remaining hyps are all invalid, we can simply skip

                    new_hyp_score = top_cand_hyp_scores[beam_id, cand_idx].item()
                    if math.isinf(new_hyp_score): break

                    prev_hyp_id = prev_hyp_ids[beam_id, cand_idx].item()
                    prev_hyp = beams[env_name][prev_hyp_id]
                    hyp_action_id = hyp_action_ids[beam_id, cand_idx].item()

                    new_hyp_env = prev_hyp['env'].clone()  # TODO: this is painfully slow
                    rel_action_id = new_hyp_env.valid_actions.index(hyp_action_id)
                    ob_t, _, _, info = new_hyp_env.step(rel_action_id)

                    if new_hyp_env.done:
                        if not new_hyp_env.error:
                            new_hyp = Hypothesis(env=new_hyp_env, score=new_hyp_score)
                            completed_hyps[new_hyp_env.name].append(new_hyp)
                    else:
                        new_hyp_beam_abs_pos = max_live_hyp_num * beam_id + prev_hyp_id
                        new_hyp = dict(env=new_hyp_env, score=new_hyp_score,
                                       ob=ob_t, parent_beam_abs_pos=new_hyp_beam_abs_pos)

                        new_beams.setdefault(env_name, []).append(new_hyp)

                        if len(new_beams.get(env_name, [])) == live_beam_size:
                            break

            if len(new_beams) == 0:
                break

            # pad the beam
            new_max_live_hyp_num = max(len(v) for v in new_beams.values())
            observations_t = []
            new_hyp_beam_abs_pos_list = []
            hyp_scores_tm1 = []
            for env_name, beam in new_beams.items():
                live_hyp_num = len(beam)
                padded_beam = beam
                if live_hyp_num < new_max_live_hyp_num:
                    padded_beam = beam + [empty_hyp] * (new_max_live_hyp_num - live_hyp_num)

                for hyp in padded_beam:
                    observations_t.append(hyp['ob'])
                    new_hyp_beam_abs_pos_list.append(hyp['parent_beam_abs_pos'])
                    hyp_scores_tm1.append(hyp['score'])

            new_hyp_state_t = [(s[0][new_hyp_beam_abs_pos_list], s[1][new_hyp_beam_abs_pos_list]) for s in state_t.state]
            new_hyp_memory_t = state_t.memory[new_hyp_beam_abs_pos_list]
            hyp_scores_tm1 = torch.tensor(hyp_scores_tm1, device=self.device)

            state_tm1 = DecoderState(state=new_hyp_state_t, memory=new_hyp_memory_t)
            observations_tm1 = observations_t
            beams = new_beams

            # compute new padded context encoding if needed
            new_live_beam_names = [env_name for env_name in beams]
            if new_live_beam_names != live_beam_names or new_max_live_hyp_num != max_live_hyp_num:
                live_beam_ids = [i for i, env in enumerate(environments) if env.name in new_beams]
                context_encoding_expanded = _expand_context(context_encoding, live_beam_ids, new_max_live_hyp_num)
            live_beam_names = new_live_beam_names
            max_live_hyp_num = new_max_live_hyp_num

        # rank completed hypothesis
        for env_name in completed_hyps.keys():
            sorted_hyps = sorted(completed_hyps[env_name], key=lambda hyp: hyp.score, reverse=True)[:beam_size]
            completed_hyps[env_name] = [Sample(trajectory=Trajectory.from_environment(hyp.env), prob=hyp.score) for hyp in sorted_hyps]

        return completed_hyps

    def decode_examples(self, environments: List[QAProgrammingEnv], beam_size, batch_size=32):
        decode_results = []

        with torch.no_grad():
            batch_iter = nn_util.batch_iter(environments, batch_size, shuffle=False)
            for batched_envs in tqdm(batch_iter, total=len(environments) // batch_size, file=sys.stdout):
                batch_decode_result = self.new_beam_search(batched_envs, beam_size=beam_size)

                batch_decode_result = list(batch_decode_result.values())
                decode_results.extend(batch_decode_result)

        return decode_results

    def sample_action(self, logits, valid_action_mask, return_log_prob=False):
        """
        logits: (batch_size, action_num)
        valid_action_mask: (batch_size, action_num)
        """

        # p_actions = nn_util.masked_softmax(logits, mask=valid_action_mask)
        logits.masked_fill_((1 - valid_action_mask).byte(), -math.inf)
        p_actions = F.softmax(logits, dim=-1)
        # (batch_size, 1)
        sampled_actions = torch.multinomial(p_actions, num_samples=1)

        if return_log_prob:
            log_p_actions = nn_util.masked_log_softmax(logits, mask=valid_action_mask)
            log_prob = torch.gather(log_p_actions, dim=1, index=sampled_actions).squeeze(-1)

            return sampled_actions.squeeze(-1), log_prob

        return sampled_actions.squeeze(-1)

    @staticmethod
    def build(config, params=None):
        encoder = Encoder.build(config, params=params)
        decoder = Decoder.build(config)

        return PGAgent(encoder, decoder, config=config)

    def save(self, model_path, kwargs=None):
        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, gpu_id=-1, **kwargs):
        device = torch.device("cuda:%d" % gpu_id if gpu_id >= 0 else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        config = params['config']
        config.update(kwargs)
        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model = PGAgent.build(config, params=params['state_dict'], **kwargs)
        model.load_state_dict(params['state_dict'])
        model = model.to(device)
        model.eval()

        return model
