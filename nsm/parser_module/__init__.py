from .agent import PGAgent
from .sketch_guided_agent import SketchGuidedAgent


def get_parser_agent_by_name(name):
    if name == 'vanilla':
        return PGAgent
    elif name == 'sketch':
        return SketchGuidedAgent
