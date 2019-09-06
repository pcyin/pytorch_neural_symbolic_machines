from typing import Union, List

import torch
from torch import nn as nn

from nsm import nn_util
from nsm.env_factory import Observation
from nsm.parser_module.bert_decoder import BertDecoder
from nsm.parser_module.bert_encoder import BertEncoder
from nsm.parser_module.decoder import MultiLayerDropoutLSTMCell, DecoderState
from nsm.parser_module.encoder import ContextEncoding
from nsm.sketch.sketch_predictor import SketchEncoding, SketchEncoder


class ContentBasedDecoder(BertDecoder):
    def __init__(self, *args, **kwargs):
        BertDecoder.__init__(self, *args, **kwargs)

        self.question_att_vec_linear = nn.Linear(self.hidden_size + self.encoder_output_size, self.hidden_size, bias=False)

        self.init_weights()

    def step(
        self,
        x: Union[List[Observation], Observation],
        state_tm1: DecoderState,
        context_encoding: ContextEncoding,
    ):
        """Perform one step of the decoder"""

        # first convert listed input to batched ones
        if isinstance(x, list):
            x = Observation.to_batched_input(x, memory_size=self.memory_size).to(self.device)

        batch_size = x.read_ind.size(0)

        # collect y_tm1 as inputs to inner rnn cells
        # Memory: (batch_size, mem_size, mem_value_dim)
        # (batch_size, mem_value_dim)
        input_mem_entry = state_tm1.memory[torch.arange(batch_size, device=self.device), x.read_ind]

        # (batch_size, hidden_size)
        inner_output_t, inner_state_t = self.rnn_cell(input_mem_entry, state_tm1.state)

        # attention over context
        ctx_t, alpha_t = self.attention_func(
            query=inner_output_t,
            keys=context_encoding['question_encoding_att_linear'],
            values=context_encoding['question_encoding'],
            entry_masks=context_encoding['question_mask']
        )

        # (batch_size, hidden_size)
        q_att_t = torch.tanh(self.question_att_vec_linear(torch.cat([inner_output_t, ctx_t], 1)))
        # att_t = self.dropout(att_t)

        # compute scores over valid memory entries
        # memory is organized by:
        # [built-in functions, constants and variables]

        # dot product attention
        # `att_t` is the query vector of cell encodings
        # table_encoding: (batch_size, row_num, column_num, cell_encoding_size)

        # (batch_size, column_num, row_num, cell_encoding_size)
        table_encoding = context_encoding['table_encoding'].permute(0, 2, 1, 3)
        table_mask = context_encoding['table_mask'].permute(0, 2, 1)

        # (batch_sie, column_num, row_num)
        cell_att_logits = torch.matmul(
            table_encoding,
            q_att_t.view(batch_size, 1, -1, 1),
        ).squeeze(-1)
        cell_att_prob = nn_util.masked_softmax(cell_att_logits, table_mask)

        # (batch_sie, column_num, 1, row_num) x (batch_size, column_num, row_num, cell_encoding_size)
        # = (batch_size, column_num, 1, cell_encoding_size)
        # (batch_size, column_num, cell_encoding_size)
        column_ctx_vec = torch.matmul(
            cell_att_prob.unsqueeze(2), table_encoding).squeeze(2)

        # (batch_size, column_num)
        column_logits = torch.matmul(column_ctx_vec, q_att_t.unsqueeze(-1)).squeeze(-1)
        column_prob = nn_util.masked_softmax(column_logits, context_encoding['column_mask'])
        # (batch_size, cell_encoding_size)
        table_ctx_vec = torch.matmul(column_prob.unsqueeze(1), column_ctx_vec).squeeze(1)

        # (batch_size, hidden_size)
        att_t = torch.tanh(self.att_vec_linear(torch.cat([q_att_t, table_ctx_vec], dim=-1)))

        # replace logic entries corresponding to columns
        memory = self.replace_column_memory_entry(state_tm1.memory, column_ctx_vec, context_encoding['column_mask'])

        # (batch_size, mem_size)
        mem_logits = torch.matmul(memory, att_t.unsqueeze(-1)).squeeze(-1)

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
        memory_tm1 = memory
        memory_t = memory_tm1.scatter_add(
            1, write_ind.view(-1, 1, 1).expand(-1, -1, memory_tm1.size(-1)),
            write_value.unsqueeze(1)
        )

        state_t = DecoderState(state=inner_state_t, memory=memory_t)

        return mem_logits, state_t

    def replace_column_memory_entry(self, memory, column_encoding, column_mask):
        # memory: (batch_size, mem_size, encoding_size)
        # column_encoding: (batch_size, column_num, encoding_size)

        column_nums = column_mask.sum(dim=-1).int().tolist()
        column_start_idx = self.builtin_func_num
        memory_copy = memory.clone()

        for e_id in range(memory.size(0)):  # over batch size
            column_num = column_nums[e_id]
            column_end_idx = column_start_idx + column_num
            memory_copy[e_id, column_start_idx: column_end_idx] = column_encoding[e_id, :column_num]

        return memory_copy
