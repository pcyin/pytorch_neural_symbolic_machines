from typing import List, Dict, Tuple, Union, Set

import json
import editdistance
import numpy as np

from nsm.agent_factory import Sample
from nsm.env_factory import QAProgrammingEnv
from nsm.program_cache import SharedProgramCache


class QuestionSimilarityModel(object):
    def __init__(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix

    @classmethod
    def load(cls, file_path):
        data = json.load(open(file_path))
        for v in data.values():
            for x in v:
                del x['question']

        return cls(data)

    def get_similar_questions(self, question_id, K):
        result = [(x['id'], x['similarity']) for x in self.similarity_matrix[question_id][:K]]

        return result


class ConsistencyModel(object):
    def __init__(self, question_similarity_model: QuestionSimilarityModel, program_cache: SharedProgramCache, K: int = 10, alpha: float = 0.03):
        self.question_similarity_model = question_similarity_model
        self.program_cache = program_cache
        self.K = K
        self.alpha = alpha

    def compute_consistency_score(self,
                                  q_id,
                                  hypotheses,
                                  K = None):
        if K is None:
            K = self.K

        nn_question_and_scores = self.question_similarity_model.get_similar_questions(q_id, K)
        nn_question_and_scores = [(q, q_sim) for q, q_sim in nn_question_and_scores if self.program_cache.contains_env(q)]
        nn_question_predictions = dict()
        for nn_qid, _ in nn_question_and_scores:
            nn_hyps = self.program_cache.get_hypotheses(nn_qid)
            nn_question_predictions[nn_qid] = nn_hyps

        supports = []
        for i, hyp_i in enumerate(hypotheses):
            support_i = 0.

            for nn_qid, q_sim in nn_question_and_scores:
                nn_hyp_probs = [e['prob'] for e in nn_question_predictions[nn_qid]]
                prob_sum = sum(nn_hyp_probs)
                normalized_nn_hyp_prob = [p / prob_sum for p in nn_hyp_probs]

                nn_hypotheses = nn_question_predictions[nn_qid]
                nn_hypotheses = sorted(nn_hypotheses, key=lambda x: -x['prob'])[:20]

                for j, nn_hyp_j in enumerate(nn_hypotheses):
                    program_sim = ConsistencyModel.compute_program_similarity(hyp_i, nn_hyp_j['program'])
                    nn_program_prob = normalized_nn_hyp_prob[j]
                    consistency_score = 10 * q_sim * program_sim  # * nn_program_prob # (1 / nn_hyp_j['norm_prob'])
                    support_i += consistency_score

            supports.append(support_i)

        return supports

    def rescore(self, log_p_samples, consistency_scores):
        """rescore each hypothesis based on consistency score"""
        if not isinstance(log_p_samples, np.ndarray):
            log_p_samples = np.array(log_p_samples)
        if not isinstance(consistency_scores, np.ndarray):
            consistency_scores = np.array(consistency_scores)

        scores = log_p_samples + self.alpha * consistency_scores
        exp_scores = np.exp(scores)
        norm_scores = exp_scores / np.sum(exp_scores)

        return norm_scores

    @staticmethod
    def compute_program_similarity(program1, program2):
        program_sig1 = ConsistencyModel.get_canonical_program_signiture(program1)
        program_sig2 = ConsistencyModel.get_canonical_program_signiture(program2)

        # print(program_sig1, program_sig2)
        edit_dist = editdistance.eval(program_sig1, program_sig2)
        sim = 1 if edit_dist == 0 else 0

        return sim

    @staticmethod
    def get_canonical_program_signiture(program):
        stack = []
        i = 0
        var_count = dict()

        while i < len(program):
            token = program[i]
            if token == '(':
                new_expr = []
                stack.append(new_expr)
            elif token == ')':
                pass
            elif token == '<END>':
                pass
            elif token.startswith('v'):
                canonical_var = var_count.setdefault(token, f'v{len(var_count)}')
                stack[-1].append(canonical_var)
            else:
                stack[-1].append(token)

            i += 1

        canonical_var_num = 0
        canonical_vars = dict()
        program_sig = []
        for expr in stack:
            for token in expr:
                # if token in var_count:
                #    token = var_count[token]

                program_sig.append(token)
        # program_sig = [expr[0] for expr in stack]

        # print(program_sig)
        return program_sig
