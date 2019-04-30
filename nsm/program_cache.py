from multiprocessing import Manager
from typing import List


class SharedProgramCache(object):
    def __init__(self):
        self.program_cache = Manager().dict()

    def add_hypothesis(self, env_name: str, program: List[str], prob: float):
        if env_name not in self.program_cache:
            self.program_cache[env_name] = dict()

        hypotheses = self.program_cache[env_name]
        hypotheses[' '.join(program)] = prob
        self.program_cache[env_name] = hypotheses

    def update_hypothesis(self, env_name: str, program: List[str], prob: float):
        hypotheses = self.program_cache[env_name]
        hypotheses[' '.join(program)] = prob
        self.program_cache[env_name] = hypotheses
