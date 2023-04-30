import json5
from src.nfa import NFA
from src.dfa_utils import nfa_to_dfa,simplify_dfa



name = '2-12.a'

with open(f'{name}.json','r') as f:
  nfa: NFA = json5.load(f, object_hook=NFA.from_dict)

graph = nfa.visualize()
graph.name = f'{name}.nfa'
graph.format = 'pdf'
graph.view(cleanup=True)

dfa = nfa_to_dfa(nfa)
graph = dfa.visualize()
graph.name = f'{name}.dfa'
graph.format = 'pdf'
graph.view(cleanup=True)

sim_dfa = simplify_dfa(dfa)
graph = sim_dfa.visualize()
graph.name = f'{name}.sim_dfa'
graph.format = 'pdf'
graph.view(cleanup=True)

