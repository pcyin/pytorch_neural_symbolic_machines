from typing import List, Dict, Tuple, Union, Set

import editdistance

from nsm.agent_factory import Sample
from nsm.env_factory import QAProgrammingEnv


def get_canonical_program_signiture(program: List[str]):
    stack = []
    i = 0
    while i < len(program):
        token = program[i]
        if token == '(':
            new_expr = []
            stack.append(new_expr)
        elif token == ')':
            pass
        elif token == '<END>':
            pass
        else:
            stack[-1].append(token)

        i += 1

    program_sig = [expr[0] for expr in stack]

    return program_sig


def compute_program_similarity(program1, program2):
    program_sig1 = get_canonical_program_signiture(program1)
    program_sig2 = get_canonical_program_signiture(program2)

    # print(program_sig1, program_sig2)
    sim = -editdistance.eval(program_sig1, program_sig2)
    return sim


def compute_consistency_score(env_name, hyp_program, nearest_neighbors, decode_results_dict, K=5):
    similar_questions = nearest_neighbors[env_name][:K]
    similar_questions = [q for q in similar_questions if q in decode_results_dict]

    support = 0.
    for nn_qid in similar_questions:
        similar_question_hyps = decode_results_dict[nn_qid]
        for nbr_hyp in similar_question_hyps:
            is_nbr_hyp_correct = nbr_hyp['is_correct'] == 1.
            if is_nbr_hyp_correct:
                similarity = compute_program_similarity(hyp_program, nbr_hyp['program'])
                similarity *= nbr_hyp.prob
                support += similarity

    return support
