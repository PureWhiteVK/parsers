from typing import Iterable, List, Any, Dict, Set, Union, cast
from graphviz import Digraph

from prettytable import PrettyTable

if __name__ == '__main__':
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import Oslash, check_type, check_array_type, State, Input, StatePair, TransPair, Epsilon, state_name


class Transition:
    # here we treat epsilon as a valid input (which we will never get it)
    def __init__(self, current: State, target: State,
                 inputs: List[Input]) -> None:
        check_type(current, State, 'Transition.current')
        check_type(target, State, 'Transition.target')
        check_array_type(inputs, Input, list, 'Transition.inputs', True)
        if len(inputs) == 1 and inputs[0] == Epsilon:
            self._inputs = []
        self._current = current
        self._target = target
        self._inputs = inputs

    @property
    def current(self):
        return self._current

    @property
    def target(self):
        return self._target

    @property
    def inputs(self):
        return self._inputs

    @property
    def is_epsilon(self):
        return len(self.inputs) == 0

    def __repr__(self) -> str:
        return f'{self.current}->{self.target} on {{{",".join(self.inputs)}}}'

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Transition):
            return False
        return self.target == __o.target and self.current == __o.current and self.inputs == __o.inputs

    def __hash__(self) -> int:
        return hash((self.target, *self.inputs, self.current))

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Transition':
        current = cast(State, data.get('current'))
        target = cast(State, data.get('target'))
        inputs = cast(List[Input], data.get('inputs'))
        return Transition(current, target, inputs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'Transition',
            'current': self.current,
            'target': self.target,
            'inputs': self.inputs
        }


class NFA:
    # a nfa should have 5 attributes
    # a state set
    # a start state
    # an accept states set
    # a transition function set
    # a input set
    def __init__(self, states: Iterable[State], start_state: State,
                 accept_states: Iterable[State], inputs: Iterable[Input],
                 transitions: Iterable[Transition]) -> None:

        self._accept_states: List[State] = list(accept_states)
        self._states: List[State] = list(states)
        self._inputs: List[Input] = list(inputs)
        self._start_state: State = start_state
        self._transitions: List[Transition] = list(transitions)

        for trans in transitions:
            if trans.is_epsilon:
                self._inputs.append(Epsilon)

        self._trans_table = NFA.trans_list_to_trans_table(transitions)

    @staticmethod
    def trans_table_to_trans_list(
            trans_table: Dict[TransPair, List[State]]) -> List[Transition]:
        trans_list: Dict[StatePair, List['Input']] = {}
        for trans_pair, target_states in trans_table.items():
            for state in target_states:
                state_pair = StatePair(trans_pair.current, state)
                if trans_list.get(state_pair) is None:
                    trans_list[state_pair] = []
                trans_list[state_pair].append(trans_pair.input)
        return [
            Transition(state_pair.current, state_pair.target, inputs)
            for state_pair, inputs in trans_list.items()
        ]

    @staticmethod
    def trans_list_to_trans_table(
            trans_list: Iterable[Transition]) -> Dict[TransPair, Set[State]]:
        trans_table: Dict[TransPair, Set[State]] = {}

        for _tran in trans_list:
            if _tran.is_epsilon:
                trans_pair = TransPair(_tran.current, Epsilon)
                if trans_table.get(trans_pair) is None:
                    trans_table[trans_pair] = set()
                trans_table[trans_pair].add(_tran.target)
            for _input in _tran.inputs:
                trans_pair = TransPair(_tran.current, _input)
                if trans_table.get(trans_pair) is None:
                    trans_table[trans_pair] = set()
                trans_table[trans_pair].add(_tran.target)
        return trans_table

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

    @property
    def start_state(self) -> State:
        return self._start_state

    def query(self, key: TransPair) -> Set[State]:
        return self._trans_table.get(key, set())

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> Union['NFA', Transition]:
        # using data['type'] to distinguish transition object or nfa object
        type_name = cast(str, data.get('type'))
        if type_name == 'Transition':
            return Transition.from_dict(data)
        elif type_name != 'NFA':
            raise ValueError(
                f'type field of json object must be `Transition` or `NFA`, requested: {type_name}'
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
        return NFA(states, start_state, accept_states, inputs, transitions)

    def to_dict(self) -> Dict[str, Any]:
        # get transition list from transition table
        return {
            'type': 'NFA',
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
                states = self.query(TransPair(s, i))
                row.append(Oslash if len(states) ==
                           0 else ','.join(map(state_name, states)))
            table.add_row(row)
        return table.get_string()

    def visualize(self) -> Digraph:
        '''
        visualize transition graph
        '''
        g = Digraph(name='nfa', graph_attr={'rankdir': 'LR'})

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
                   ','.join(trans.inputs),
                   arrowsize='0.5')
        return g


def main():

    def t(f: State, t: State, i: List[Input]) -> Transition:
        return Transition(f, t, i)

    s0 = 'q0'
    s1 = 'q1'
    s2 = 'q2'
    s3 = 'q3'
    Q = [s0, s1, s2, s3]
    q0 = s0
    F = [s3]
    a = 'a'
    b = 'b'
    Sigma = [a, b]
    Delta = [t(s0, s0, [a, b]), t(s0, s1, [a]), t(s1, s2, [b]), t(s2, s3, [b])]
    nfa = NFA(Q, q0, F, Sigma, Delta)
    print('*' * 40)
    print('nfa:')
    print(nfa)
    g = nfa.visualize()
    g.view(cleanup=True)
    nfa_json = nfa.to_dict()
    import json
    from src.utils import OUTPUT_DIR
    import os
    new_nfa: NFA = json.loads(json.dumps(nfa_json), object_hook=nfa.from_dict)
    with open(os.path.join(OUTPUT_DIR, 'output_nfa.json'),
              'w',
              encoding='utf-8') as f:
        json.dump(new_nfa.to_dict(), f)
    print('*' * 40)
    print('new nfa:')
    print(new_nfa)


if __name__ == '__main__':
    main()