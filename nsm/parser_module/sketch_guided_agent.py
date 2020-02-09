import collections
import math
import sys
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch
from torch import nn as nn

import nsm.execution.worlds.wikitablequestions
from nsm import nn_util, data_utils
from nsm.execution import executor_factory
from nsm.parser_module.agent import PGAgent
from nsm.computer_factory import SPECIAL_TKS
from nsm.env_factory import Trajectory, Observation, Sample
from nsm.parser_module.bert_encoder import BertEncoder
from nsm.parser_module.decoder import DecoderBase
from nsm.parser_module.encoder import EncoderBase
from nsm.parser_module.sketch_guided_decoder import SketchGuidedDecoder
from nsm.sketch.sketch import Sketch
from nsm.sketch.sketch_predictor import SketchPredictor
from nsm.sketch.sketch_encoder import SketchEncoder


class SketchGuidedAgent(PGAgent):
    "Agent trained by policy gradient."

    def __init__(
        self,
        encoder: EncoderBase, decoder: DecoderBase,
        sketch_predictor: SketchPredictor,
        sketch_encoder: SketchEncoder,
        config: Dict, discount_factor: float = 1.0,
        log=None
    ):
        super(SketchGuidedAgent, self).__init__(encoder, decoder, sketch_predictor, config)

        self.config = config
        self.discount_factor = discount_factor

        self.encoder = encoder
        self.decoder = decoder

        self.sketch_predictor = sketch_predictor
        self.sketch_encoder = sketch_encoder

        self.log = log

    def compute_trajectory_actions_prob(self, trajectories: List[Trajectory], return_info=False) -> torch.Tensor:
        contexts = [traj.context for traj in trajectories]
        sketches = [Sketch(traj.program) for traj in trajectories]

        sketch_prob = self.sketch_predictor(
            contexts,
            sketches
        )

        context_encoding = self.encoder.encode(contexts)

        batched_observation_seq, tgt_actions_info = Trajectory.to_batched_sequence_tensors(
            trajectories, self.memory_size)

        sketch_encoding = self.sketch_encoder(sketches)

        state_tm1 = init_state = self.decoder.get_initial_state(context_encoding, sketch_encoding)

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
            mem_logits, state_t = self.decoder.step(
                obs_slice_t,
                state_tm1,
                context_encoding,
                sketch_encoding
            )

            action_logits.append(mem_logits)
            state_tm1 = state_t

        # (max_action_len, batch_size, memory_size)
        action_logits = torch.stack(action_logits, dim=0).permute(1, 0, 2)

        # (batch_size, max_action_len, memory_size)
        action_log_probs = nn_util.masked_log_softmax(action_logits, batched_observation_seq.valid_action_mask)

        # (batch_size, max_action_len)
        tgt_action_log_probs = torch.gather(action_log_probs, dim=-1, index=tgt_action_id.unsqueeze(-1)).squeeze(-1) * tgt_action_mask

        # (batch_size, max_action_len)
        variable_ground_mask = sketch_encoding['var_time_step_mask']
        tgt_variable_grounding_prob = tgt_action_log_probs * variable_ground_mask

        # (batch_size)
        tgt_trajectory_log_probs = sketch_prob + tgt_variable_grounding_prob.sum(dim=-1)

        if return_info:
            info = dict(
                action_log_probs=action_log_probs,
                tgt_action_id=tgt_action_id,
                tgt_action_mask=tgt_action_mask,
                action_logits=action_logits,
                valid_action_mask=batched_observation_seq.valid_action_mask,
                context_encoding=context_encoding
            )

            return tgt_trajectory_log_probs, info

        return tgt_trajectory_log_probs

    def sample(
        self, environments, sample_num, use_cache=False,
        constraint_sketches: Dict = None,
    ):
        if sample_num == 0:
            return []

        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        batch_env_sketches = self.sketch_predictor.get_sketches(
            environments,
            K=5
        )

        if self.log:
            print('*' * 10, 'SAMPLE BEGIN', '*' * 10, file=self.log)

        sketches = []
        duplicated_envs = []

        for env_idx, env in enumerate(environments):
            env_sketches = batch_env_sketches[env_idx]
            if not env_sketches:
                continue

            sample_temp = self.config.get('sketch_sample_temperature', 1.)
            env_probs = np.exp([sketch.prob / sample_temp for sketch in env_sketches])
            env_probs /= env_probs.sum()

            sampled_sketch_indices = np.random.choice(
                list(range(len(env_sketches))),
                size=sample_num, replace=True,
                p=env_probs
            )
            sampled_sketches = [env_sketches[idx] for idx in sampled_sketch_indices]

            if self.log:
                print(f"Question [{env.name}]: {env.question_annotation['question']} ({len(env_sketches)} Sketches)",
                      file=self.log)
                for sketch in env_sketches:
                    print(sketch, file=self.log)

                print('Sampled Sketches:', file=self.log)
                for sketch in sampled_sketches:
                    print(sketch, file=self.log)

            sketches.extend(sampled_sketches)
            duplicated_envs.extend([
                env.clone()
                for _ in range(len(sampled_sketches))
            ])

        if not duplicated_envs:
            return []

        environments = duplicated_envs
        for env in environments:
            env.use_cache = use_cache

        env_context = [env.get_context() for env in environments]
        context_encoding = self.encode(env_context)
        sketch_encoding = self.sketch_encoder(sketches)

        completed_envs = []
        active_envs = environments

        observations_tm1 = [env.start_ob for env in environments]
        state_tm1 = self.decoder.get_initial_state(context_encoding, sketch_encoding)
        sample_probs = torch.zeros(len(environments), device=self.device)

        while True:
            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(self.device)
            mem_logits, state_t = self.decoder.step(
                observations_tm1,
                state_tm1,
                context_encoding,
                sketch_encoding
            )

            # try:
            # (batch_size)
            sampled_action_t_id, sampled_action_t_prob = self.sample_action(
                mem_logits, batched_ob_tm1.valid_action_mask,
                return_log_prob=True)
            # except RuntimeError:
            #     for ob in observations_tm1:
            #         print(f'Observation {ob}', file=sys.stderr)
            #         print(ob.valid_action_indices, file=sys.stderr)
            #
            #     print(batched_ob_tm1.valid_action_mask, file=sys.stderr)
            #     torch.save((mem_logits, batched_ob_tm1.valid_action_mask), 'tmp.bin')
            #     exit(-1)

            variable_mask = sketch_encoding['var_time_step_mask'][:, state_tm1.t]
            sample_probs = sample_probs + sampled_action_t_prob * variable_mask

            observations_t = []
            new_active_env_pos = []
            new_active_envs = []
            has_completed_sample = False
            for env_id, (env, action_t) in enumerate(zip(active_envs, sampled_action_t_id.tolist())):
                sketch = sketches[env_id]

                if state_t.t - 1 >= len(sketch):
                    print('Sketch: ', sketch, file=sys.stderr)
                    print('t=', state_tm1.t, file=sys.stderr)
                    print('Index Error!', file=sys.stderr)
                    raise ValueError()

                sketch_token_tm1 = sketch[state_t.t - 1]
                if not sketch.is_variable_slot(sketch_token_tm1):
                    # ues sketch's predefined action instead of the sampled one
                    sketch_action_t = env.de_vocab.lookup(sketch_token_tm1)
                    # if sketch token not in valid continuating actions
                    # if sketch_action_t in env.valid_actions:

                        # print('Sketch: ', sketch, file=sys.stderr)
                        # print('t =', state_tm1.t, 'token = ', sketch_token_tm1, file=sys.stderr)
                        # print('Sketch Action', env.de_vocab.lookup(sketch_action_t, reverse=True), file=sys.stderr)
                        # print('Valid Actions: ', env.de_vocab.lookup(env.valid_actions, reverse=True), file=sys.stderr)
                        # raise RuntimeError()

                    action_t = sketch_action_t

                if action_t in env.valid_actions:
                    action_rel_id = env.valid_actions.index(action_t)
                    ob_t, _, _, info = env.step(action_rel_id)
                    if env.done:
                        completed_envs.append((
                            env,
                            sketch.prob + sample_probs[env_id].item()
                        ))
                        has_completed_sample = True
                    else:
                        observations_t.append(ob_t)
                        new_active_env_pos.append(env_id)
                        new_active_envs.append(env)
                else:
                    assert sketch_action_t not in env.valid_actions
                    has_completed_sample = True

            if not new_active_env_pos:
                break

            if has_completed_sample:
                # need to perform slicing
                for key in ['question_encoding', 'question_mask', 'question_encoding_att_linear']:
                    context_encoding[key] = context_encoding[key][new_active_env_pos]
                for key in ['var_time_step_mask', 'value']:
                    sketch_encoding[key] = sketch_encoding[key][new_active_env_pos]

                state_tm1 = state_t[new_active_env_pos]
                sample_probs = sample_probs[new_active_env_pos]
                sketches = [sketches[i] for i in new_active_env_pos]
            else:
                state_tm1 = state_t

            observations_tm1 = observations_t
            active_envs = new_active_envs

        # if self.log:
        #     print("Samples:", file=self.log)

        samples = []
        for env_id, (env, prob) in enumerate(completed_envs):
            if not env.error:
                traj = Trajectory.from_environment(env)
                samples.append(Sample(trajectory=traj, prob=prob))

                # if self.log:
                #     print(f"{' '.join(traj.human_readable_program)} (correct={traj.reward == 1.}, prob={prob})", file=self.log)

        return samples

    def new_beam_search(self, environments, beam_size, use_cache=False, return_list=False,
                        constraint_sketches=None, strict_constraint_on_sketches=False, force_sketch_coverage=False):
        # if already explored everything, then don't explore this environment anymore.
        if use_cache:
            # if already explored everything, then don't explore this environment anymore.
            environments = [env for env in environments if not env.cache.is_full()]

        for env in environments:
            env.use_cache = use_cache

        Hypothesis = collections.namedtuple('Hypothesis', ['sketch', 'env', 'score'])

        CandidateHyp = collections.namedtuple(
            'CandidateHyp',
            ['sketch', 'prev_hyp_env', 'action_id', 'rel_action_id', 'score', 'prev_hyp_abs_pos', 'human_action_token']
        )

        def _expand_encoding(_tensor_dict, indices, keys=None):
            if keys is None:
                keys = [
                    key for key, val in _tensor_dict.items()
                    if isinstance(val, torch.Tensor) and val.dim() > 1
                ]

            for key in keys:
                sliced_tensor = _tensor_dict[key][indices]
                _tensor_dict[key] = sliced_tensor

            return _tensor_dict

        beams = OrderedDict()
        completed_hyps = OrderedDict((env.name, []) for env in environments)

        # (env_num, ...)
        env_context = [env.get_context() for env in environments]
        context_encoding = self.encode(env_context)

        # List[List * env_num]
        nested_hyp_sketches = []
        if constraint_sketches:
            # print('decoding using predefined sketches', file=sys.stdout)
            for env in environments:
                nested_hyp_sketches.append(
                    constraint_sketches.get(env.name, []))
        else:
            nested_hyp_sketches = self.sketch_predictor.get_sketches(environments, K=5)

        if self.log:
            print(f"Beam Search for questions:", file=self.log)

        hyp_sketches = []  # flatten the hyp sketch list
        flattened_hyp_env_idx_ptr = []
        hyp_num = 0
        for i, (env, env_hyp_sketches) in enumerate(
                zip(environments, nested_hyp_sketches)):
            env_sketch_num = len(env_hyp_sketches)
            hyp_num += env_sketch_num
            hyp_sketches.extend(env_hyp_sketches)

            # clone each environment while adding them to the initial beam
            beams[env.name] = [
                # hypothesis is initialized with sketch's probability
                Hypothesis(sketch=sketch, env=env.clone(), score=sketch.prob)
                for sketch
                in env_hyp_sketches
            ]

            flattened_hyp_env_idx_ptr.extend([i] * env_sketch_num)

            if self.log:
                print(f"[{env.name}] {env.question_annotation['question']}", file=self.log)

        def _log_beam(_beams):
            print("Current Beam Configuration:", file=self.log)
            for env_name, beam in _beams.items():
                print(f"======[{env_name}]======", file=self.log)
                for hyp in beam:
                    program = hyp.env.to_human_readable_program()
                    print(f"sketch={hyp.sketch} program={program} (score={hyp.score})", file=self.log)

        if self.log:
            print(f"Initial Beam", file=self.log)
            _log_beam(beams)

        if hyp_num == 0:
            return [] if return_list else completed_hyps

        sketch_encoding_expanded = sketch_encoding = self.sketch_encoder(hyp_sketches)

        # (hyp_num, xxx)
        observations_tm1 = [
            hyp.env.start_ob
            for env_name, beam in beams.items()
            for hyp in beam
        ]

        context_encoding_expanded = _expand_encoding(
            context_encoding,
            indices=flattened_hyp_env_idx_ptr
        )

        state_tm1 = self.decoder.get_initial_state(context_encoding_expanded, sketch_encoding_expanded)
        hyp_scores_tm1 = torch.tensor(
            [
                hyp.score
                for beam in beams.values()
                for hyp in beam
            ], device=self.device
        )

        while beams:
            if self.log:
                print(f't={state_tm1.t}', file=self.log)

            batched_ob_tm1 = Observation.to_batched_input(observations_tm1, memory_size=self.memory_size).to(self.device)

            # (hyp_num, memory_size)
            action_probs_t, state_t = self.decoder.step_and_get_action_scores_t(
                batched_ob_tm1, state_tm1,
                context_encoding=context_encoding_expanded,
                sketch_encoding=sketch_encoding_expanded
            )

            # no need to -inf over invalid slots, since we will only enumerate over
            # valid entries later
            # action_probs_t[(1 - batched_ob_tm1.valid_action_mask).byte()] = float('-inf')

            variable_slot_mask = sketch_encoding['var_time_step_mask'][:, state_tm1.t]
            action_probs_t = action_probs_t * variable_slot_mask.unsqueeze(-1)

            # (hyp_num, memory_size)
            cont_cand_hyp_scores = action_probs_t + hyp_scores_tm1.unsqueeze(-1)

            # collect continuating candidates for new hypotheses
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

                if self.log:
                    print(f"Question[{env_name}] {live_beam_size} living hyps", file=self.log)

                for prev_hyp_id, prev_hyp in enumerate(beam):
                    hyp_sketch = prev_hyp.sketch
                    sketch_token = hyp_sketch[state_tm1.t]

                    if self.log:
                        print(f"\tHyp: sketch={prev_hyp.sketch} program={prev_hyp.env.program} Sketch token={sketch_token} Score={prev_hyp.score}", file=self.log)

                    # if it is a variable grounding step
                    if hyp_sketch.is_variable_slot(sketch_token):
                        if self.log:
                            print(f"\tvariable grounding", file=self.log)

                        valid_action_indices = prev_hyp.env.valid_actions
                        _cont_action_scores = beam_new_cont_scores[prev_hyp_id][
                            valid_action_indices].cpu()

                        for rel_action_id, new_hyp_score in enumerate(_cont_action_scores):
                            abs_action_id = prev_hyp.env.obs[-1].valid_action_indices[rel_action_id]
                            new_hyp_score = new_hyp_score.item()
                            if not math.isinf(new_hyp_score):
                                if self.log:
                                    action_token = prev_hyp.env.de_vocab.lookup(abs_action_id, reverse=True)
                                    human_readable_token = prev_hyp.env.get_human_readable_action_token(action_token)
                                else:
                                    human_readable_token = None

                                candidate_hyp = CandidateHyp(
                                    sketch=prev_hyp.sketch,
                                    prev_hyp_env=prev_hyp.env,
                                    rel_action_id=rel_action_id,
                                    action_id=abs_action_id,
                                    score=new_hyp_score,
                                    prev_hyp_abs_pos=beam_start + prev_hyp_id,
                                    human_action_token=human_readable_token
                                )

                                if self.log:
                                    print(f"\t\tvar={candidate_hyp.human_action_token} align score={new_hyp_score - prev_hyp.score}", file=self.log)

                                continuing_candidates[env_name].append(candidate_hyp)
                    else:
                        # if it is an idle run step (encode sketch token)
                        abs_action_id = prev_hyp.env.de_vocab.lookup(sketch_token)
                        valid_action_indices = prev_hyp.env.valid_actions
                        if abs_action_id in valid_action_indices:
                            rel_action_id = valid_action_indices.index(abs_action_id)

                            candidate_hyp = CandidateHyp(
                                sketch=prev_hyp.sketch,
                                prev_hyp_env=prev_hyp.env,
                                rel_action_id=rel_action_id,
                                action_id=abs_action_id,
                                score=prev_hyp.score,
                                prev_hyp_abs_pos=beam_start + prev_hyp_id,
                                human_action_token=sketch_token
                            )
                            continuing_candidates[env_name].append(candidate_hyp)

                            if self.log:
                                print(f"\t\tIdle run, use sketch token", file=self.log)

                # rank all hypotheses together with completed ones
                all_candidates = completed_hyps[env_name] + continuing_candidates[env_name]
                all_candidates.sort(key=lambda hyp: hyp.score, reverse=True)

                if self.log:
                    print(f"Ranked hypothesis:", file=self.log)
                    for hyp in all_candidates:
                        if isinstance(hyp, Hypothesis):
                            print("finished hyp", hyp.env.program, hyp.score, file=self.log)
                        else:
                            env = hyp.prev_hyp_env
                            print(f"sketch={hyp.sketch} "
                                  f"program={env.to_human_readable_program() + [hyp.human_action_token]} "
                                  f"score={hyp.score}", file=self.log)

                # top_k_candidates = heapq.nlargest(beam_size, all_candidates, key=lambda x: x.score)
                completed_hyps[env_name] = []

                def _add_hypothesis_to_new_beam(_hyp):
                    if isinstance(_hyp, Hypothesis):
                        completed_hyps[env_name].append(_hyp)
                    else:
                        new_hyp_env = _hyp.prev_hyp_env.clone()

                        ob_t, _, _, info = new_hyp_env.step(_hyp.rel_action_id)

                        if new_hyp_env.done:
                            if not new_hyp_env.error:
                                new_hyp = Hypothesis(env=new_hyp_env, score=_hyp.score, sketch=_hyp.sketch)
                                completed_hyps[new_hyp_env.name].append(new_hyp)
                        else:
                            new_hyp = Hypothesis(env=new_hyp_env, score=_hyp.score, sketch=_hyp.sketch)
                            new_beams.setdefault(env_name, []).append(new_hyp)

                            new_hyp_parent_abs_pos_list.append(_hyp.prev_hyp_abs_pos)
                            observations_t.append(ob_t)
                            new_hyp_scores.append(_hyp.score)

                new_beam_size = 0
                if force_sketch_coverage:
                    env_new_beam_not_covered_sketches = set(hyp.sketch for hyp in beam)

                for cand_hyp in all_candidates:
                    if new_beam_size < beam_size:
                        _add_hypothesis_to_new_beam(cand_hyp)

                        if force_sketch_coverage:
                            cand_hyp_covered_sketches = set(
                                sketch
                                for sketch
                                in env_new_beam_not_covered_sketches
                                if sketch == cand_hyp.sketch
                            )
                            env_new_beam_not_covered_sketches -= cand_hyp_covered_sketches

                    # make sure each sketch has at least one candidate hypothesis in the new beam
                    elif force_sketch_coverage and env_new_beam_not_covered_sketches:
                        cand_hyp_covered_sketches = set(
                            sketch
                            for sketch
                            in env_new_beam_not_covered_sketches
                            if sketch == cand_hyp.sketch
                        )

                        if cand_hyp_covered_sketches:
                            _add_hypothesis_to_new_beam(cand_hyp)
                            env_new_beam_not_covered_sketches -= cand_hyp_covered_sketches

                    new_beam_size += 1

                beam_start = beam_end

            if len(new_beams) == 0:
                break

            if self.log:
                _log_beam(new_beams)

            state_tm1 = state_t[new_hyp_parent_abs_pos_list]
            observations_tm1 = observations_t
            hyp_scores_tm1 = torch.tensor(new_hyp_scores, device=self.device)

            _expand_encoding(
                context_encoding_expanded,
                new_hyp_parent_abs_pos_list,
                ['question_encoding', 'question_mask', 'question_encoding_att_linear'],
            )
            _expand_encoding(
                sketch_encoding_expanded,
                new_hyp_parent_abs_pos_list,
                ['value', 'mask', 'var_time_step_mask']
            )

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
                samples = [Sample(trajectory=Trajectory.from_environment(hyp.env), prob=hyp.score, sketch=hyp.sketch) for hyp in _hyps]
                samples_list.extend(samples)

            return samples_list

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

        encoder = BertEncoder.build(config)

        # FIXME: hacky!
        if config.get('use_trainable_sketch_manager', False):
            sketch_predictor = SketchPredictor.build(config, encoder=encoder)
            sketch_encoder = SketchEncoder.build(config, sketch_predictor)
        else:
            sketch_predictor = sketch_encoder = None

        decoder = SketchGuidedDecoder.build(config, encoder, sketch_encoder)

        return cls(
            encoder, decoder,
            sketch_predictor=sketch_predictor,
            sketch_encoder=sketch_encoder,
            config=config
        )

    def save(self, model_path, kwargs=None):
        ddp = None
        if isinstance(self.encoder.bert_model, nn.DataParallel):
            ddp = self.encoder.bert_model
            self.encoder.bert_model = ddp.module

        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        if ddp:
            self.encoder.bert_model = ddp

        torch.save(params, model_path)

    @staticmethod
    def load(model_path, default_values_handle=None, gpu_id=-1, **kwargs):
        device = torch.device("cuda:%d" % gpu_id if gpu_id >= 0 else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        config = params['config']

        if default_values_handle:
            default_values_handle(config)

        config.update(kwargs)
        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model = PGAgent.build(config, params=params['state_dict'], **kwargs)
        incompatible_keys = model.load_state_dict(params['state_dict'], strict=False)
        if incompatible_keys.missing_keys:
            print('Loading agent, got missing keys {}'.format(incompatible_keys.missing_keys), file=sys.stderr)
        if incompatible_keys.unexpected_keys:
            print('Loading agent, got unexpected keys {}'.format(incompatible_keys.unexpected_keys), file=sys.stderr)

        model = model.to(device)
        model.eval()

        return model