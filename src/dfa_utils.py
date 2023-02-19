from typing import List, Optional, Tuple, Union, overload, Set, Dict
from typing_extensions import TypeAlias
from prettytable import PrettyTable

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import State, Input, is_array_type, Epsilon, TransPair, NameAllocator, Oslash, state_name
from src.dfa import DFA, Transition as DFA_Trans
from src.nfa import NFA

StateClosure: TypeAlias = Tuple[State, ...]

GroupId: TypeAlias = int


def nfa_to_dfa(nfa: NFA) -> DFA:
    """
    convert nfa to dfa using subset construction algorithm
    """

    @overload
    def epsilon_closure(state: State) -> StateClosure:
        ...

    @overload
    def epsilon_closure(state: StateClosure) -> StateClosure:
        ...

    def epsilon_closure(
            state: Union[State, Tuple[State, ...]]) -> Tuple[State, ...]:
        if isinstance(state, State):
            return epsilon_closure(tuple([state]))
        elif is_array_type(state, State, tuple):
            # actually this is a simple graph traversal algorithm (BFS)
            queue = list(s for s in state)
            closure = set(state)
            while len(queue) != 0:
                curr_state = queue.pop(0)
                # 从 curr_state 出发可以直接到达的 state
                for next_state in move(curr_state, Epsilon):
                    if not next_state in closure:
                        closure.add(next_state)
                        queue.append(next_state)
            return tuple(closure)
        raise TypeError(
            f'epsilon_closure(...) just support Tuple[int,...] or int, requested: {state}'
        )

    @overload
    def move(_state: State, _input: Input) -> StateClosure:
        ...

    @overload
    def move(_state: StateClosure, _input: Input) -> StateClosure:
        ...

    def move(_state: Union[StateClosure, State],
             _input: Input) -> StateClosure:
        if isinstance(_state, State):
            return tuple(nfa.query(TransPair(_state, _input)))
        elif is_array_type(_state, State, tuple):
            res: Set[State] = set()
            for state in _state:
                res.update(move(state, _input))
            return tuple(res)
        raise TypeError(
            f'move(...) just support Tuple[State] or State, requested: {_state},{_input}'
        )

    state_allocator = NameAllocator()
    closure_state_map: Dict[Tuple[State, ...], State] = dict()
    inputs = nfa.inputs.copy()
    # no epsilon in dfa
    if Epsilon in inputs:
        inputs.remove(Epsilon)

    initial_closure = epsilon_closure(nfa.start_state)
    # ε-closure -> new-state-id mapping
    start_state = state_allocator.next
    closure_state_map[initial_closure] = start_state

    stack: List[StateClosure] = [initial_closure]

    transitions: Set[DFA_Trans] = set()

    while len(stack) != 0:
        curr = stack.pop(-1)
        for _input in inputs:
            target = epsilon_closure(move(curr, _input))
            if len(target) == 0:
                continue
            if not target in closure_state_map.keys():
                closure_state_map[target] = state_allocator.next
                stack.append(target)
            _current = closure_state_map[curr]
            _target = closure_state_map[target]
            transitions.add(DFA_Trans(_current, _target, _input))

    accept_states = [
        dfa_state for nfa_states, dfa_state in closure_state_map.items()
        if any(s in nfa.accept_states for s in nfa_states)
    ]

    states = state_allocator.names

    dfa: DFA = DFA(states, start_state, accept_states, inputs, transitions)

    table = PrettyTable(['NFA STATE', 'DFA STATE', *inputs])
    for nfa_states, dfa_state in closure_state_map.items():
        row = [f'{{{",".join(map(state_name,nfa_states))}}}', dfa_state]
        for _input in inputs:
            target = dfa.query(TransPair(dfa_state, _input))
            row.append(Oslash if target is None else target)
        table.add_row(row)
    print(table)

    return dfa


def simplify_dfa(dfa: DFA) -> DFA:
    # initial state, we got to state groups, 0 indicates the accept states, 1 indicates the unaccept states
    state_group_map: Dict[State, GroupId] = dict(
        (s, 0 if s in dfa.accept_states else 1) for s in dfa.states)
    # 对于状态-1而言，其所属的group为-1（也是一个虚拟的group，不属于任何的group）
    # the virtual_state means the fallback state
    FALLBACK_STATE = f'q{len(dfa.states)}'
    FALLBACK_GROUP_ID: GroupId = -1
    state_group_map[FALLBACK_STATE] = FALLBACK_GROUP_ID

    group_count = 2

    def groups() -> Dict[GroupId, List[State]]:
        res: Dict[GroupId, List[State]] = {}
        for state, group_id in state_group_map.items():
            if res.get(group_id) is None:
                res[group_id] = list()
            res[group_id].append(state)
        return res

    def print_groups(groups: Dict[GroupId, List[State]]) -> None:
        table = PrettyTable(['GroupId', 'States'])
        table.add_rows([f'{group_id}', ','.join(map(state_name, states))]
                       for group_id, states in groups.items()
                       if group_id != FALLBACK_GROUP_ID)
        print(table)

    def subdivide_group(group_id: GroupId, group: List[State]) -> bool:
        # 不对-1状态进行划分，我们也没办法划分
        if group_id == FALLBACK_GROUP_ID:
            return False
        # this will modify state_map (generate new groups)
        nonlocal group_count, state_group_map
        for _input in dfa.inputs:
            target_group_map: Dict[GroupId, List[State]] = {}
            for state in group:
                target = dfa.query(TransPair(state, _input))
                target_group = FALLBACK_GROUP_ID if target is None else state_group_map[
                    target]
                if target_group_map.get(target_group) is None:
                    target_group_map[target_group] = list()
                target_group_map[target_group].append(state)
            if len(target_group_map) != 1:
                print(
                    f'subdive group: {",".join(map(state_name,group))} on input: {_input}'
                )
                print(
                    f'subdive result:\b {"/".join(",".join(map(state_name,s)) for s in target_group_map.values())}'
                )
                # perform subdivide
                for states in target_group_map.values():
                    new_group_id = group_count
                    group_count += 1
                    for s in states:
                        state_group_map[s] = new_group_id
                return True
        return False

    n_iter = 0
    changed = True
    while changed:
        changed = any(
            subdivide_group(group_id, group)
            for group_id, group in groups().items())
        n_iter += 1

    final_groups = groups()

    # perform constructing...
    start_group_id: GroupId = -1
    # find start state
    name_allocator = NameAllocator()
    new_state_group_map: Dict[State, GroupId] = {}
    group_new_state_map: Dict[GroupId, State] = {}
    for group_id, group in final_groups.items():
        if dfa.start_state in group:
            start_group_id = group_id
            break

    s = name_allocator.next
    new_state_group_map[s] = start_group_id
    group_new_state_map[start_group_id] = s

    for group_id in final_groups.keys():
        if group_id == start_group_id or group_id == FALLBACK_GROUP_ID:
            continue
        s = name_allocator.next
        new_state_group_map[s] = group_id
        group_new_state_map[group_id] = s

    states = name_allocator.names

    if len(states) == len(dfa.states):
        print('dfa is already simplified!')
        return dfa

    new_state_group_map = dict((v, k) for k, v in group_new_state_map.items())

    transitions: Set[DFA_Trans] = set()
    for trans in dfa.transitions:
        current = group_new_state_map[state_group_map[trans.current]]
        target = group_new_state_map[state_group_map[trans.target]]
        transitions.add(DFA_Trans(current, target, trans.input))

    accept_states: List[State] = [
        group_new_state_map[state_group_map[state]]
        for state in dfa.accept_states
    ]

    start_state: State = group_new_state_map[state_group_map[dfa.start_state]]

    sim_dfa: DFA = DFA(states, start_state, accept_states, dfa.inputs,
                       transitions)

    table = PrettyTable(['DFA STATE', 'New DFA STATE', *sim_dfa.inputs])
    for new_state in sim_dfa.states:
        row = [
            f'{{{",".join(map(state_name,final_groups[new_state_group_map[new_state]]))}}}',
            new_state
        ]
        for _input in sim_dfa.inputs:
            _target = sim_dfa.query(TransPair(new_state, _input))
            row.append(Oslash if _target is None else _target)
        table.add_row(row)
    print(table)
    return sim_dfa


def main():
    nfa: Optional[NFA] = None
    import json
    with open('test_nfa.json', 'r', encoding='utf-8') as f:
        nfa = json.load(f, object_hook=NFA.from_dict)
    if nfa is None:
        exit(-1)
    g = nfa.visualize()
    g.view(cleanup=True)
    dfa = nfa_to_dfa(nfa)
    print(dfa)
    g = dfa.visualize()
    g.view(cleanup=True)
    new_dfa = simplify_dfa(dfa)
    g = new_dfa.visualize()
    g.view(cleanup=True)


if __name__ == '__main__':
    main()