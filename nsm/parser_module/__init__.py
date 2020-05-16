from .agent import PGAgent
from .sketch_guided_agent import SketchGuidedAgent
# from nsm.parser_module.content_based_agent import ContentBasedAgent


def get_parser_agent_by_name(name):
    if name == 'vanilla':
        return PGAgent
    elif name == 'sketch':
        return SketchGuidedAgent
    # elif name == 'content_based':
    #     return ContentBasedAgent
    else:
        raise ValueError(name)
