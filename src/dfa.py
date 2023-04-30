from typing import Iterable, List, Any, Dict, Optional, Union, cast
from graphviz import Digraph
from functools import cached_property
from prettytable import PrettyTable

if __name__ == '__main__':
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import Oslash, check_type, check_array_type, State, Input, TransPair, Epsilon, state_name


class Transition:
    def __init__(self, current: State, target: State, input: Input) -> None:
        check_type(current, State, 'Transition.current')
        check_type(target, State, 'Transition.target')
        check_type(input, Input, 'Transition.input')
        assert input != Epsilon, f'DFA does not allow {Epsilon}-transition!'
        self._current = current
        self._target = target
        self._input = input

    @property
    def current(self):
        return self._current

    @property
    def target(self):
        return self._target

    @property
    def input(self):
        return self._input

    def __repr__(self) -> str:
        return f'{self.current}->{self.target} on {",".join(self.input)}'

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Transition):
            return False
        return self.target == __o.target and self.current == __o.current and self.input == __o.input

    def __hash__(self) -> int:
        return hash((self.target, self.input, self.current))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Transition':
        current = cast(State, data.get('current'))
        target = cast(State, data.get('target'))
        input = cast(Input, data.get('input'))
        return Transition(current, target, input)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'Transition',
            'current': self.current,
            'target': self.target,
            'input': self.input
        }


class DFA:
    def __init__(self, states: Iterable[State], start_state: State,
                 accept_states: Iterable[State], inputs: Iterable[Input],
                 transitions: Iterable[Transition]) -> None:

        self._accept_states: List[State] = list(accept_states)
        self._states: List[State] = list(states)
        self._inputs: List[Input] = list(inputs)
        self._start_state: State = start_state
        self._transitions: List[Transition] = list(transitions)

        self._trans_table: Dict[TransPair, State] = dict([
            (TransPair(trans.current, trans.input), trans.target)
            for trans in transitions
        ])

    @property
    def transitions(self) -> List[Transition]:
        return self._transitions

    @property
    def inputs(self) -> List[Input]:
        return self._inputs

    @property
    def states(self) -> List[State]:
        return self._states

    @property
    def accept_states(self) -> List[State]:
        return self._accept_states

    @cached_property
    def non_accpet_states(self) -> List[State]:
        return [s for s in self.states if s not in self.accept_states]

    @property
    def start_state(self) -> State:
        return self._start_state

    def query(self, key: TransPair) -> Optional[State]:
        return self._trans_table.get(key)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Union['DFA', Transition]:
        # using data['type'] to distinguish Transition object or DFA object
        type_name = cast(str, data.get('type'))
        if type_name == 'Transition':
            return Transition.from_dict(data)
        elif type_name != 'DFA':
            raise ValueError(
                f'type field must be `Transition` or `DFA`, requested: {type_name}'
            )
        states = cast(List[State], data.get('states'))
        start_state = cast(State, data.get('start_state'))
        accept_states = cast(List[State], data.get('accept_states'))
        inputs = cast(List[Input], data.get('inputs'))
        transitions = cast(List[Transition], data.get('transitions'))
        check_type(start_state, State, 'start_state')
        check_array_type(states, State, list, 'states')
        check_array_type(accept_states, State, list, 'accept_states')
        check_array_type(inputs, Input, list, 'inputs')
        check_array_type(transitions, Transition, list, 'transitions')
        return DFA(states, start_state, accept_states, inputs, transitions)

    def to_dict(self) -> Dict[str, Any]:
        # get transition list from transition table
        return {
            'type': 'DFA',
            'states': list(self.states),
            'start_state': self.start_state,
            'accept_states': list(self.accept_states),
            'inputs': list(self.inputs),
            'transitions': [trans.to_dict() for trans in self.transitions]
        }

    def __repr__(self) -> str:
        table = PrettyTable(['STATE', *self.inputs])
        # right alignment
        table.align['STATE'] = 'r'

        def format_state(s: State) -> str:
            res = f'{s}'
            if s == self.start_state:
                res = f'-> {res}'
            if s in self.accept_states:
                res = f'* {res}'
            return res

        for s in self.states:
            row: List[str] = [format_state(s)]
            for i in self.inputs:
                t = self.query(TransPair(s, i))
                row.append(Oslash if t is None else state_name(t))
            table.add_row(row)
        return table.get_string()

    def visualize(self) -> Digraph:
        '''
        visualize transition graph
        '''
        g = Digraph(name='dfa', graph_attr={'rankdir': 'LR'})

        g.node(name='vnode', label='', shape='none')

        for state in self.states:
            name = state_name(state)
            if state in self.accept_states:
                g.node(name=name, label=name, shape='doublecircle')
            else:
                g.node(name=name, label=name, shape='circle')

        g.edge('vnode',
               state_name(self.start_state),
               label='start',
               arrowsize='0.5')

        for trans in self.transitions:
            g.edge(state_name(trans.current),
                   state_name(trans.target),
                   trans.input,
                   arrowsize='0.5')
        return g


def main():
    def t(f: State, t: State, i: Input) -> Transition:
        return Transition(f, t, i)

    s0 = 'q0'
    s1 = 'q1'
    s2 = 'q2'
    s3 = 'q3'
    s4 = 'q4'
    Q = [s0, s1, s2, s3, s4]
    q0 = s0
    F = [s4]
    a = 'a'
    b = 'b'
    Sigma = [a, b]
    Delta = [
        t(s0, s1, a),
        t(s0, s2, b),
        t(s1, s1, a),
        t(s1, s3, b),
        t(s2, s1, a),
        t(s2, s2, b),
        t(s3, s1, a),
        t(s3, s4, b),
        t(s4, s1, a),
        t(s4, s2, b)
    ]
    dfa = DFA(Q, q0, F, Sigma, Delta)
    print('*' * 40)
    print('dfa:')
    print(dfa)
    g = dfa.visualize()
    g.view(cleanup=True)
    nfa_json = dfa.to_dict()
    import json
    from src.utils import OUTPUT_DIR
    import os
    new_dfa: DFA = json.loads(json.dumps(nfa_json), object_hook=dfa.from_dict)
    with open(os.path.join(OUTPUT_DIR, 'output_dfa.json'),
              'w',
              encoding='utf-8') as f:
        json.dump(new_dfa.to_dict(), f)
    print('*' * 40)
    print('new nfa:')
    print(new_dfa)


if __name__ == '__main__':
    main()