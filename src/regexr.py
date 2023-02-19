# convert the parse tree of regular expression to dfa
from typing import Iterable, List, Dict, Optional
from typing_extensions import TypeAlias
import json

if __name__ == '__main__':
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.parse_tree import ParseTree
from src.nfa import NFA, Transition
from src.utils import State, Input, Epsilon, NameAllocator
from src.dfa_utils import nfa_to_dfa, simplify_dfa

StateId: TypeAlias = int

INTERNAL_NODE_LABEL = '$r'


class RegExrNFA(NFA):

    def __init__(self, states: Iterable[State], start_state: State,
                 accept_state: State, inputs: Iterable[Input],
                 transitions: Iterable[Transition]) -> None:
        super().__init__(states, start_state, [accept_state], inputs,
                         transitions)
        self._accept_state = list(self.accept_states)[0]

    @property
    def accept_state(self) -> State:
        return self._accept_state


def alphabet_exr_nfa(input: Input) -> RegExrNFA:
    return RegExrNFA(['0', '1'], '0', '1', [input], [Transition('0', '1', [input])])


def epsilon_exr_nfa() -> RegExrNFA:
    return alphabet_exr_nfa(Epsilon)


def rename(r: RegExrNFA, alloc: NameAllocator) -> Dict[State, State]:
    name: Dict[State, State] = {}
    name[r.start_state] = alloc.next
    for s in r.states:
        if s == r.start_state or s == r.accept_state:
            continue
        name[s] = alloc.next
    name[r.accept_state] = alloc.next
    return name


def union(r1: RegExrNFA, r2: RegExrNFA) -> RegExrNFA:
    inputs = set([*r1.inputs, *r2.inputs, Epsilon])
    transitions: List[Transition] = []
    name_allocator = NameAllocator()

    start_state = name_allocator.next

    r1_name: Dict[State, State] = rename(r1, name_allocator)
    r2_name: Dict[State, State] = rename(r2, name_allocator)

    accept_state = name_allocator.next

    r1_start_state = r1_name[r1.start_state]
    r1_accept_state = r1_name[r1.accept_state]

    r2_start_state = r2_name[r2.start_state]
    r2_accept_state = r2_name[r2.accept_state]

    for trans in r1.transitions:
        c = r1_name[trans.current]
        t = r1_name[trans.target]
        transitions.append(Transition(c, t, trans.inputs))

    for trans in r2.transitions:
        c = r2_name[trans.current]
        t = r2_name[trans.target]
        transitions.append(Transition(c, t, trans.inputs))

    transitions.append(Transition(start_state, r1_start_state, [Epsilon]))
    transitions.append(Transition(start_state, r2_start_state, [Epsilon]))
    transitions.append(Transition(r1_accept_state, accept_state, [Epsilon]))
    transitions.append(Transition(r2_accept_state, accept_state, [Epsilon]))

    states = name_allocator.names
    return RegExrNFA(states, start_state, accept_state, inputs, transitions)


def concat(r1: RegExrNFA, r2: RegExrNFA) -> RegExrNFA:
    inputs = set([*r1.inputs, *r2.inputs, Epsilon])
    transitions: List[Transition] = []

    name_allocator = NameAllocator()

    r1_name: Dict[State, State] = {}
    r2_name: Dict[State, State] = {}

    r1_name[r1.start_state] = name_allocator.next
    for s in r1.states:
        # we have to make sure start_state is q0 here
        if s == r1.start_state or s == r1.accept_state:
            continue
        r1_name[s] = name_allocator.next
    r1_name[r1.accept_state] = name_allocator.next
    r2_name[r2.start_state] = r1_name[r1.accept_state]
    for s in r2.states:
        if s == r2.start_state or s == r2.accept_state:
            continue
        r2_name[s] = name_allocator.next
    r2_name[r2.accept_state] = name_allocator.next

    for trans in r1.transitions:
        c = r1_name[trans.current]
        t = r1_name[trans.target]
        transitions.append(Transition(c, t, trans.inputs))

    for trans in r2.transitions:
        c = r2_name[trans.current]
        t = r2_name[trans.target]
        transitions.append(Transition(c, t, trans.inputs))

    start_state = r1_name[r1.start_state]
    accept_state = r2_name[r2.accept_state]

    states = name_allocator.names

    return RegExrNFA(states, start_state, accept_state, inputs, transitions)


def closure(r: RegExrNFA) -> RegExrNFA:
    inputs = set([*r.inputs, Epsilon])
    transitions: List[Transition] = []

    name_allocator = NameAllocator()

    start_state = name_allocator.next

    name: Dict[State, State] = rename(r, name_allocator)

    accept_state = name_allocator.next

    states = name_allocator.names

    for trans in r.transitions:
        c = name[trans.current]
        t = name[trans.target]
        transitions.append(Transition(c, t, trans.inputs))

    r_start_state = name[r.start_state]
    r_accept_state = name[r.accept_state]

    transitions.append(Transition(start_state, r_start_state, [Epsilon]))
    transitions.append(Transition(start_state, accept_state, [Epsilon]))
    transitions.append(Transition(r_accept_state, r_start_state, [Epsilon]))
    transitions.append(Transition(r_accept_state, accept_state, [Epsilon]))

    return RegExrNFA(states, start_state, accept_state, inputs, transitions)


def reg_exr_parse_tree_to_nfa(tree: ParseTree) -> RegExrNFA:

    class ParseError(RuntimeError):

        def __init__(self, *args: object) -> None:
            super().__init__(*args)

    def is_leaf_node(node: ParseTree.Node) -> bool:
        return node.label != INTERNAL_NODE_LABEL

    def is_interior_node(node: ParseTree.Node) -> bool:
        return node.label == INTERNAL_NODE_LABEL

    def leaf_equals(node: ParseTree.Node, ch: str) -> bool:
        return is_leaf_node(node) and node.label == ch

    def recursive(node: ParseTree.Node) -> RegExrNFA:
        # 根据子节点类型判断产生式
        """
        r -> rr (concat)
        r -> r* (kleene closure)
        r -> r|r (union)
        r -> (r) (forward)
        r -> char (char)
        r -> epsilon (epsilon)
        """
        c = node._children
        r: Optional[RegExrNFA] = None
        if len(c) == 3:
            if is_interior_node(c[0]) and is_interior_node(
                    c[2]) and leaf_equals(c[1], '|'):
                r1 = recursive(c[0])
                r2 = recursive(c[2])
                r = union(r1, r2)
            elif leaf_equals(c[0], '(') and is_interior_node(
                    c[1]) and leaf_equals(c[2], ')'):
                r = recursive(c[1])
        elif len(node._children) == 2:
            if is_interior_node(c[0]) and is_interior_node(c[1]):
                r1 = recursive(c[0])
                r2 = recursive(c[1])
                r = concat(r1, r2)
            elif is_interior_node(c[0]) and leaf_equals(c[1], '*'):
                r0 = recursive(c[0])
                r = closure(r0)
        elif len(node._children) == 1:
            if leaf_equals(c[0], Epsilon):
                r = epsilon_exr_nfa()
            elif is_leaf_node(c[0]):
                r = alphabet_exr_nfa(c[0].label)
        if r is None:
            raise ParseError('not a valid RegExr ParseTree')
        return r

    r = recursive(tree.root)
    if r == None:
        raise ParseError('not a valid RegExr ParseTree')
    return r


def main():

    def l(ch: str) -> ParseTree.Node:
        return ParseTree.Node(ch)

    def i(*children: ParseTree.Node) -> ParseTree.Node:
        return ParseTree.Node(INTERNAL_NODE_LABEL, list(children))

    l0 = l('(')
    l1 = l('a')
    l2 = l('|')
    l3 = l('b')
    l4 = l(')')
    l5 = l('*')
    l6 = l('a')
    l7 = l('b')
    l8 = l('b')
    n0 = i(l1)
    n1 = i(l3)
    n2 = i(n0, l2, n1)
    n3 = i(l0, n2, l4)
    n4 = i(n3, l5)
    n5 = i(l6)
    n6 = i(n4, n5)
    n7 = i(l7)
    n8 = i(n6, n7)
    n9 = i(l8)
    n10 = i(n8, n9)

    tree = ParseTree(n10)

    with open('test_regexr.json', 'w') as f:
        json.dump(tree.to_dict(), f)

    g = tree.visualize()
    g.view(cleanup=True)
    nfa = reg_exr_parse_tree_to_nfa(tree)
    g = nfa.visualize()
    g.view(cleanup=True)
    dfa = nfa_to_dfa(nfa)
    sim_dfa = simplify_dfa(dfa)
    g = sim_dfa.visualize()
    g.view(cleanup=True)


if __name__ == '__main__':
    main()
