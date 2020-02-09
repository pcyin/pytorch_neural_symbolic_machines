from typing import Dict, Optional

import nsm.execution.worlds.wikitablequestions
from nsm import data_utils
from nsm.execution import executor_factory
from nsm.computer_factory import SPECIAL_TKS
from nsm.parser_module import PGAgent
from nsm.parser_module.content_based_decoder import ContentBasedDecoder
from nsm.parser_module.content_based_encoder import ContentBasedEncoder
from nsm.parser_module.decoder import DecoderBase
from nsm.parser_module.encoder import EncoderBase
from nsm.sketch.sketch_predictor import SketchPredictor


class ContentBasedAgent(PGAgent):
    "Agent trained by policy gradient."

    def __init__(
        self,
        encoder: EncoderBase, decoder: DecoderBase,
        sketch_predictor: Optional[SketchPredictor],
        config: Dict
    ):
        super(PGAgent, self).__init__()

        self.config = config

        self.encoder = encoder
        self.decoder = decoder
        self.sketch_predictor = sketch_predictor

    @classmethod
    def build(cls, config, params=None):
        dummy_kg = {
            'kg': None,
            'num_props': [],
            'datetime_props': [],
            'props': [],
            'row_ents': []
        }

        executor = nsm.execution.worlds.wikitablequestions.WikiTableExecutor(dummy_kg)
        api = executor.get_api()
        op_vocab = data_utils.Vocab(
            [f['name'] for f in api['func_dict'].values()] +
            ['all_rows'] +
            SPECIAL_TKS
        )
        config['builtin_func_num'] = op_vocab.size

        encoder = ContentBasedEncoder.build(config)
        decoder = ContentBasedDecoder.build(config, encoder)

        return cls(
            encoder, decoder,
            sketch_predictor=None,
            config=config
        )

    @property
    def sufficient_context_encoding_entries(self):
        fields = [
            'question_encoding', 'question_mask', 'question_encoding_att_linear',
            'table_encoding', 'table_mask', 'column_mask'
        ]

        return fields
