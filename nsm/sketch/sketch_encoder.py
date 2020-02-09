from typing import Dict, List

import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nsm.sketch.sketch import Sketch
from nsm.sketch.sketch_predictor import SketchPredictor


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