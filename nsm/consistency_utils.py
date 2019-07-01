from typing import List, Dict, Tuple, Union, Set

import json
import editdistance
import numpy as np
import sys

import torch

from nsm.env_factory import QAProgrammingEnv, Sample
from nsm.program_cache import SharedProgramCache


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))

    return e_x / e_x.sum()


class QuestionSimilarityModel(object):
    def __init__(self, similarity_matrix):
        self.similarity_matrix = similarity_matrix

    @classmethod
    def load(cls, file_path):
        data = torch.load(file_path)['similar_questions']
        # for v in data.values():
        #     for x in v:
        #         del x['question']

        return cls(data)

    def get_similar_questions(self, question_id, K):
        result = [x for x in self.similarity_matrix[question_id][:K]]

        return result


class ConsistencyModel(object):
    def __init__(self, question_similarity_model: QuestionSimilarityModel, program_cache: SharedProgramCache, environments: List[QAProgrammingEnv],
                 K: int = 10, alpha: float = 0.01, log_file: str = 'consistency.log', debug: bool = False):
        self.question_similarity_model = question_similarity_model
        self.program_cache = program_cache
        self.K = K
        self.alpha = alpha

        self.environments = {e.name: e for e in environments}
        self.log_file = open(log_file, 'w')
        self.debug = debug

    def compute_consistency_score(self,
                                  q_id: str,
                                  hypotheses: List[Sample],
                                  K = None):
        if K is None:
            K = self.K

        nn_questions = self.question_similarity_model.get_similar_questions(q_id, K)
        nn_questions = [q for q in nn_questions if self.program_cache.contains_env(q['id'])]
        nn_question_predictions = dict()
        for nn_question in nn_questions:
            nn_hyps = self.program_cache.get_hypotheses(nn_question['id'])
            nn_question_predictions[nn_question['id']] = nn_hyps

        debug = self.debug
        if debug:
            f_log = self.log_file
            print('', file=f_log)
            print('==============================================', file=f_log)
            print(f'Id: {q_id}', file=f_log)
            print(f'Question: %s' % self.environments[q_id].question_annotation['question'], file=f_log)
            print('', file=f_log)
            print('==============Similiar Questions==============', file=f_log)

            for nn_question in nn_questions:
                print(f'Question[{nn_question["id"]}]: {nn_question["question"]} ||| similarity={nn_question["similarity"]}', file=f_log)

            print('', file=f_log)
            print('', file=f_log)
            print('', file=f_log)

        supports = []
        for i, hyp_i in enumerate(hypotheses):
            support_i = 0.

            if debug:
                program = hyp_i.trajectory.program
                sketch = self.get_canonical_program_signiture(program)
                program = ' '.join(program)
                sketch = ' '.join(sketch)
                prob = hyp_i.prob
                print('=' * 15 + f'Prediction[{i}]' + '=' * 15, file=f_log)
                print(f'Prediction[{i}]: {program} ||| prob={prob}', file=f_log)
                print(f'Sketch: {sketch}', file=f_log)
                print('', file=f_log)
                print('', file=f_log)

            for nn_question in nn_questions:
                nn_qid = nn_question['id']
                q_sim = nn_question['similarity']

                nn_hyp_probs = [e['prob'] for e in nn_question_predictions[nn_qid]]
                prob_sum = sum(nn_hyp_probs)
                normalized_nn_hyp_prob = [p / prob_sum for p in nn_hyp_probs]

                nn_hypotheses = nn_question_predictions[nn_qid]
                nn_hypotheses = sorted(nn_hypotheses, key=lambda x: -x['prob'])[:5]

                if debug:
                    similar_question = nn_question['question']
                    print(f'>>> Similiar Question[{nn_qid}]: {similar_question}', file=f_log)

                for j, nn_hyp_j in enumerate(nn_hypotheses):
                    program_sim = ConsistencyModel.compute_program_similarity(hyp_i.trajectory.program, nn_hyp_j['program'])
                    nn_program_prob = normalized_nn_hyp_prob[j]
                    consistency_score = q_sim * program_sim * nn_program_prob # (1 / nn_hyp_j['norm_prob'])
                    support_i += consistency_score

                    if debug and program_sim > 0:
                        nn_program = nn_hyp_j['program']
                        nn_program = ' '.join(nn_program)
                        print(f'Similiar Program {j}: {nn_program}', file=f_log)

                        sketch = self.get_canonical_program_signiture(nn_hyp_j['program'])
                        sketch = ' '.join(sketch)
                        print(f'Similiar Sketch: {sketch}', file=f_log)
                        print(f'Question Similarity: {q_sim}', file=f_log)
                        print(f'Program Similarity: {program_sim}', file=f_log)
                        print(f'Similiar Program Prob: {nn_program_prob}', file=f_log)
                        print(f'Consistency score from this program: {consistency_score}', file=f_log)
                        print(f'', file=f_log)
                        print(f'', file=f_log)

            supports.append(support_i)

            if debug:
                print(f'Prediction[{i}] total consistency received={support_i}', file=f_log)
                print(f'', file=f_log)
                print(f'', file=f_log)

        if debug:
            print('=' * 30, file=f_log)
            print(f'Consistency Socores for All Predictions: %s' % supports, file=f_log)

        return supports

    def rescore(self, log_p_samples, consistency_scores, alpha):
        """rescore each hypothesis based on consistency score"""
        if not isinstance(log_p_samples, np.ndarray):
            log_p_samples = np.array(log_p_samples)
        if not isinstance(consistency_scores, np.ndarray):
            consistency_scores = np.array(consistency_scores)

        p_consistency = softmax(alpha * consistency_scores)
        log_p_consistency = np.log(p_consistency)
        scores = log_p_samples + log_p_consistency

        # scores = log_p_samples + self.alpha * consistency_scores
        exp_scores = np.exp(scores)
        norm_scores = exp_scores / np.sum(exp_scores)

        return norm_scores

    def compute_consistency_and_rescore(self, q_id: str, hypotheses: List[Sample], K = None):
        consistency_scores = self.compute_consistency_score(q_id, hypotheses, K)
        consistency_scores = np.array(consistency_scores)

        log_p_samples = np.log([sample.prob for sample in hypotheses])
        p_samples = self.rescore(log_p_samples, consistency_scores, self.alpha)

        if self.debug:
            print(f'Original sample probs: %s' % log_p_samples, file=self.log_file)
            print(f'Sample probs after consistency reranking: %s' % p_samples, file=self.log_file)

        return p_samples

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
