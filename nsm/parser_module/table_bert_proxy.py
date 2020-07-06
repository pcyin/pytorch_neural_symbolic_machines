import multiprocessing
import sys
import time
from types import SimpleNamespace
import numpy as np

import torch
import torch.nn as nn
from typing import Any, Optional

from pytorch_pretrained_bert import BertTokenizer
from table_bert.config import TableBertConfig, BERT_CONFIGS

from nsm.actor import Actor
from nsm.parser_module.table_bert_helper import get_table_bert_model


class TableBertProxy(nn.Module):
    def __init__(self, actor_id: str, table_bert_config: TableBertConfig):
        super(TableBertProxy, self).__init__()
        self.request_queue = None
        self.result_queue = None

        self.actor_id = actor_id
        self.worker_id = actor_id

        self.device = torch.device('cpu')

        self.config = table_bert_config
        self.bert_config = BERT_CONFIGS[table_bert_config.base_model_name]
        self.tokenizer = BertTokenizer.from_pretrained(table_bert_config.base_model_name)

    def initialize(self, actor: Actor):
        self.request_queue = actor.table_bert_request_queue
        self.result_queue = actor.table_bert_result_queue
        self.actor = actor

    @property
    def output_size(self):
        return self.bert_config.hidden_size

    def to(self, *args, **kwargs):
        device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

        self.device = device

    @property
    def is_initialized(self):
        return self.result_queue is not None and self.request_queue is not None

    def encode(self, contexts, tables):
        assert self.is_initialized

        payload = (contexts, tables)
        request = {
            'worker_id': self.worker_id,
            'model_ver': self.actor.model_path,
            'payload': payload
        }
        self.request_queue.put(request)

        # blocking until we got the results
        encode_result = self.result_queue.get()
        unpacked_encode_result = self.unpack_encode_result(encode_result)

        return unpacked_encode_result

    def unpack_encode_result(self, result):
        def _to_pytorch_tensor(obj):
            if isinstance(obj, tuple):
                return tuple(_to_pytorch_tensor(x) for x in obj)
            elif isinstance(obj, list):
                return list(_to_pytorch_tensor(x) for x in obj)
            elif isinstance(obj, dict):
                return {
                    key: _to_pytorch_tensor(val)
                    for key, val
                    in obj.items()
                }
            elif isinstance(obj, np.ndarray):
                return torch.from_numpy(obj).to(self.device)
            else:
                return obj

        unpacked_result = _to_pytorch_tensor(result)

        return unpacked_result


class TableBertServer(multiprocessing.Process):
    def __init__(self, config: Any, device: torch.device = 'cpu'):
        super(TableBertServer, self).__init__(daemon=True)

        self.request_queue = multiprocessing.Queue()
        self.workers = dict()
        self.config = config
        self.target_device = device

        self.model_path: Optional[str] = None
        self.learner_msg_val: multiprocessing.Value = None

    @property
    def device(self):
        return next(self.table_bert.parameters()).device

    def register_worker(self, actor: Actor):
        table_bert_result_queue = getattr(actor, 'table_bert_result_queue', None)
        if not table_bert_result_queue:
            table_bert_result_queue = multiprocessing.Queue()
            setattr(actor, 'table_bert_result_queue', table_bert_result_queue)

        self.workers[actor.actor_id] = SimpleNamespace(
            result_queue=table_bert_result_queue
        )
        setattr(actor, 'table_bert_request_queue', self.request_queue)

    def init_server(self):
        target_device = self.target_device
        if 'cuda' in str(target_device):
            torch.cuda.set_device(target_device)

        self.table_bert = get_table_bert_model(
            self.config, use_proxy=False,
            master='table_bert_server'
        ).to(target_device).eval()

    def run(self):
        print('[TableBertServer] Init table bert...', file=sys.stderr)
        self.init_server()
        print('[TableBertServer] Init success', file=sys.stderr)

        cum_request_num = 0.
        cum_model_ver_not_match_num = 0.
        cum_process_time = 0.
        with torch.no_grad():
            while True:
                request = self.request_queue.get()

                cum_request_num += 1.

                payload = request['payload']
                worker_id = request['worker_id']

                worker_model_ver = request['model_ver']
                self_model_ver = self.model_path

                # if worker_model_ver != self_model_ver:
                #     self.check_and_load_new_model()

                self_model_ver = self.model_path
                if worker_model_ver != self_model_ver:
                    cum_model_ver_not_match_num += 1.
                    if cum_model_ver_not_match_num % 100 == 0:
                        print(f'[TableBertServer] Server model version does not match '
                              f'with source {self_model_ver}!={worker_model_ver}, '
                              f'ratio={cum_model_ver_not_match_num / cum_request_num}',
                              file=sys.stderr)

                t1 = time.time()
                encode_result = self.table_bert.encode(*payload)
                packed_result = self.pack_encode_result(encode_result)
                t2 = time.time()

                self.workers[worker_id].result_queue.put(packed_result)

                cum_process_time += t2 - t1
                if cum_request_num % 100 == 0:
                    print(f'cum. request={cum_request_num}, speed={cum_request_num / cum_process_time} requests/s',
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
            state_dict = {
                k.partition('encoder.bert_model.')[2]: v
                for k, v
                in state_dict.items()
                if k.startswith('encoder.bert_model.')
            }
            # print(list(state_dict.keys()))

            self.table_bert.load_state_dict(state_dict)
            self.model_path = new_model_path
            self.table_bert.eval()

            t2 = time.time()
            print('[TableBertServer] loaded new model [%s] (took %.2f s)' % (new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False
