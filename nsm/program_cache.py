from multiprocessing import Manager
from typing import List


class SharedProgramCache(object):
    def __init__(self):
        self.program_cache = Manager().dict()

    def add_hypothesis(self, env_name: str, program: List[str], prob: float):
        if env_name not in self.program_cache:
            self.program_cache[env_name] = dict()

        hypotheses = self.program_cache[env_name]
        hypotheses[' '.join(program)] = {'program': program, 'prob': prob}
        self.program_cache[env_name] = hypotheses

    def update_hypothesis(self, env_name: str, program: List[str], prob: float):
        hypotheses = self.program_cache[env_name]
        hypotheses[' '.join(program)] = {'program': program, 'prob': prob}
        self.program_cache[env_name] = hypotheses

    def contains_env(self, env_name):
        return env_name in self.program_cache

    def get_hypotheses(self, env_name):
        result = self.program_cache[env_name]
        result = [x for x in result.values() if x['prob'] is not None]
        result = sorted(result, key=lambda x: -x['prob'])

        return result

    def stat(self):
        num_envs = len(self.program_cache)

        return {'env_num': num_envs}