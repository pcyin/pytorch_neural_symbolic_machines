from multiprocessing import Manager
from typing import Any

from nsm.consistency_utils import QuestionSimilarityModel
from nsm.program_cache import SharedProgramCache


class Sketch(object):
    def __init__(self, program=None, tokens=None, prob=None):
        if program:
            self.tokens = Sketch.program_to_sketch(program)
        else:
            assert tokens
            self.tokens = tokens

        self._sketch_str = ' '.join(self.tokens)
        self.operators = self.get_operators(program)
        self.prob = prob

    @staticmethod
    def program_to_sketch(program):
        sketch_tokens = []
        for token in program:
            if token.startswith('v') or token == 'all_rows':
                token = 'v'

            sketch_tokens.append(token)

        return sketch_tokens

    def __getitem__(self, item):
        return self.tokens[item]

    def __len__(self):
        return len(self.tokens)

    def __hash__(self):
        return hash(self._sketch_str)

    def __repr__(self):
        return self._sketch_str + (f' (prob={self.prob})'
                                   if self.prob is not None
                                   else '')

    def __eq__(self, other):
        if not isinstance(other, Sketch):
            return False

        return self._sketch_str == other._sketch_str

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def get_operators(program):
        stack = []
        for token in program:
            if token == '(':
                new_expr = []
                stack.append(new_expr)
            elif token == ')':
                pass
            elif token == '<END>':
                pass
            else:
                stack[-1].append(token)

        operators = [exp[0] for exp in stack]
        return operators

    def is_compatible_with_hypothesis(self, hypothesis: Any):
        if hasattr(hypothesis, 'prev_hyp_env'):
            traj_action_ids = hypothesis.prev_hyp_env.mapped_actions + [hypothesis.action_id]
            env = hypothesis.prev_hyp_env
        else:
            assert hasattr(hypothesis, 'env')  # it's a completed hypothesis
            traj_action_ids = hypothesis.env.mapped_actions
            env = hypothesis.env

        hyp_program = env.de_vocab.lookup(traj_action_ids, reverse=True)

        return self.is_compatible_with_program(hyp_program)

    def is_compatible_with_program(self, program):
        is_compatible = True
        for p_token, s_token in zip(program, self.tokens):
            if p_token.startswith('v') or p_token == 'all_rows':
                p_token = 'v'

            if p_token != s_token:
                is_compatible = False
                break

        return is_compatible

    @staticmethod
    def is_variable_slot(token):
        return token == 'v'

    __str__ = __repr__


class SketchManager(object):
    def __init__(
        self,
        program_cache: SharedProgramCache,
        question_similarity_model: QuestionSimilarityModel
    ):
        self.program_cache = program_cache
        self.question_similarity_model = question_similarity_model

    def get_sketches_from_similar_questions(self, env_name: str, K=10, remove_explored=True, log_file=None):
        nn_questions = self.question_similarity_model.get_similar_questions(env_name, K)
        nn_questions = [
            q
            for q
            in nn_questions
            if self.program_cache.contains_env(q['id'])
        ]

        def _normalize_probs(_hyps):
            prob_sum = sum(e['prob'] for e in _hyps) + 1e-9
            for _hyp in _hyps:
                _hyp['prob'] /= prob_sum

        hyp_explored_sketches = set()
        if remove_explored:
            hypotheses = self.program_cache.get_hypotheses(env_name)
            hyp_explored_sketches = set(
                Sketch(hyp['program']) for hyp in hypotheses
            )

            if log_file:
                print(f"> Sketches that has been explored: ", file=log_file)
                for sketch in hyp_explored_sketches:
                    print(str(sketch), file=log_file)

        nn_sketches = dict()
        for nn_question in nn_questions:
            nn_hyps = self.program_cache.get_hypotheses(nn_question['id'])
            _normalize_probs(nn_hyps)
            for hyp in nn_hyps:
                hyp_program = hyp['program']
                hyp_sketch = Sketch(hyp_program)
                if hyp_sketch not in hyp_explored_sketches:
                    nn_sketches.setdefault(hyp_sketch, []).append(
                        {
                            'hyp_prob': hyp['prob'],
                            'question_id': nn_question['id'],
                            'question_similarity': nn_question['similarity']
                        }
                    )

        nn_sketch_summary = dict()
        for sketch, entries in nn_sketches.items():
            sketch_score = 0.

            for entry in entries:
                sketch_score += entry['hyp_prob'] * entry['question_similarity']

            nn_sketch_summary[sketch] = {
                'entries': entries,
                'score': sketch_score
            }

        return nn_sketch_summary
