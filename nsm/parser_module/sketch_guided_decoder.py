from typing import Union, List

import torch
from torch import nn as nn

from nsm import nn_util
from nsm.env_factory import Observation
from nsm.parser_module.bert_decoder import BertDecoder
from nsm.parser_module.bert_encoder import BertEncoder
from nsm.parser_module.decoder import MultiLayerDropoutLSTMCell, DecoderState
from nsm.parser_module.encoder import ContextEncoding
from nsm.sketch.sketch_predictor import SketchEncoding
from nsm.sketch.sketch_encoder import SketchEncoder


class SketchGuidedDecoderState(DecoderState):
    def __init__(self, state, memory, t):
        DecoderState.__init__(self, state, memory)
        self.t = t

    def __getitem__(self, indices):
        state = DecoderState.__getitem__(self, indices)

        return __class__(state.state, state.memory, self.t)


class SketchGuidedDecoder(BertDecoder):
    def __init__(self, *args, **kwargs):
        sketch_encoding_size = kwargs.pop('sketch_encoding_size')

        BertDecoder.__init__(self, *args, **kwargs)

        self.rnn_cell = MultiLayerDropoutLSTMCell(
            self.mem_item_embed_size + sketch_encoding_size, kwargs['hidden_size'],
            num_layers=kwargs['num_layers'], dropout=kwargs['dropout'])

        self.decoder_cell_init_linear = nn.Linear(
            sketch_encoding_size + self.decoder_cell_init_linear.in_features, self.hidden_size)

        self.init_weights()

    @classmethod
    def build(cls, config, encoder: BertEncoder, sketch_encoder: SketchEncoder) -> 'SketchGuidedDecoder':
        return cls(
            mem_item_embed_size=config['value_embedding_size'],
            constant_value_embed_size=config['value_embedding_size'],
            encoder_output_size=config['hidden_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['n_layers'],
            output_feature_num=config['n_de_output_features'],
            builtin_func_num=config['builtin_func_num'],
            memory_size=config['memory_size'],
            encoder=encoder,
            dropout=config['dropout'],
            sketch_encoding_size=sketch_encoder.output_size
        )

    def get_lstm_init_state(self, context_encoding: ContextEncoding, sketch_encoding: SketchEncoding):
        # use encoding of the [CLS] token to initialize the decoder
        question_repr = context_encoding['cls_encoding']
        sketch_repr = sketch_encoding['last_state']
        x = torch.cat([question_repr, sketch_repr], dim=-1)

        sc_0_i = self.decoder_cell_init_linear(x)
        sh_0_i = torch.tanh(sc_0_i)

        decoder_init_states = [(sh_0_i, sc_0_i)] * self.rnn_cell.num_layers

        return decoder_init_states

    def get_initial_state(self, context_encoding: ContextEncoding, sketch_encoding: SketchEncoding):
        # prepare decoder's initial memory and internal LSTM state

        initial_memory = self.get_initial_memory(context_encoding)
        decoder_init_states = self.get_lstm_init_state(context_encoding, sketch_encoding)

        state = SketchGuidedDecoderState(state=decoder_init_states, memory=initial_memory, t=0)

        return state

    def step(
        self,
        x: Union[List[Observation], Observation],
        state_tm1: SketchGuidedDecoderState,
        context_encoding: ContextEncoding,
        sketch_encoding: SketchEncoding
    ):
        # first convert listed input to batched ones
        if isinstance(x, list):
            x = Observation.to_batched_input(x, memory_size=self.memory_size).to(self.device)

        batch_size = x.read_ind.size(0)

        # (batch_size, sketch_token_encoding_size)
        sketch_token_encoding_t = sketch_encoding['value'][:, state_tm1.t, :]
        # (batch_size, memory_encoding_size)
        # collect y_tm1 as inputs to inner rnn cells
        # Memory: (batch_size, mem_size, mem_value_dim)
        program_token_encoding_tm1 = state_tm1.memory[torch.arange(batch_size, device=self.device), x.read_ind]

        rnn_input_t = torch.cat(
            [program_token_encoding_tm1, sketch_token_encoding_t],
            dim=-1
        )

        # (batch_size, hidden_size)
        inner_output_t, inner_state_t = self.rnn_cell(rnn_input_t, state_tm1.state)

        # attention over question encoding
        ctx_t, alpha_t = self.attention_func(
            query=inner_output_t,
            keys=context_encoding['question_encoding_att_linear'],
            values=context_encoding['question_encoding'],
            entry_masks=context_encoding['question_mask']
        )

        # (batch_size, hidden_size)
        att_t = torch.tanh(
            self.att_vec_linear(
                torch.cat([inner_output_t, ctx_t], dim=1)
            )
        )

        # compute scores over valid memory entries
        # memory is organized by:
        # [built-in functions, constants and variables]

        # dot product attention
        # (batch_size, mem_size)
        mem_logits = torch.matmul(state_tm1.memory, att_t.unsqueeze(-1)).squeeze(-1)

        # add output features to logits
        # (batch_size, mem_size)
        if self.output_feature_num:
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
        memory_t = memory_tm1.scatter_add(
            dim=1,
            index=write_ind.view(-1, 1, 1).expand(-1, -1, memory_tm1.size(-1)),
            src=write_value.unsqueeze(1)
        )

        state_t = SketchGuidedDecoderState(
            state=inner_state_t,
            memory=memory_t,
            t=state_tm1.t + 1
        )

        return mem_logits, state_t

    def step_and_get_action_scores_t(self, observations_t, state_tm1, context_encoding, sketch_encoding):
        mem_logits, state_t = self.step(
            observations_t, state_tm1,
            context_encoding=context_encoding, sketch_encoding=sketch_encoding
        )

        # (batch_size, mem_size)
        action_score_t = nn_util.masked_log_softmax(mem_logits, mask=observations_t.valid_action_mask)

        return action_score_t, state_t