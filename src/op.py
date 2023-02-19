from functools import cache
from typing import Any, List, Dict, NamedTuple, Optional, Set, Tuple, NamedTuple
from prettytable import PrettyTable
from enum import Enum, unique

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.cfg import CFG, Production, Symbol, Terminal, Nonterminal
from src.utils import Dollar
from src.cfg_parser import CfgParser

FirstVTs = Dict[Nonterminal, Set[Terminal]]
LastVTs = Dict[Nonterminal, Set[Terminal]]


@unique
class Precedence(Enum):
    EQUAL = '='
    LESS = '<'
    GREATER = '>'
    NONE = ''


@unique
class Action(Enum):
    ACCEPT = 'accept'
    SHIFT = 'shift'
    REDUCE = 'reduce'


class Key(NamedTuple):
    p0: Terminal
    p1: Terminal


def firstvt(cfg: CFG) -> FirstVTs:
    firstvts: FirstVTs = dict([(s, set()) for s in cfg.nonterminals])
    stack: List[Tuple[Nonterminal, Terminal]] = []

    def insert(P: Nonterminal, a: Terminal):
        if a not in firstvts[P]:
            firstvts[P].add(a)
            stack.append((P, a))

    for p in cfg.productions:
        P = p.head
        if p.is_epsilon:
            continue
        # P -> a...
        if cfg.is_terminal(p.body[0]):
            insert(P, p.body[0])
        # P -> Q a ...
        elif len(p.body) > 1 and cfg.is_nonterminal(
                p.body[0]) and cfg.is_terminal(p.body[1]):
            insert(P, p.body[1])
    while len(stack) != 0:
        Q, a = stack.pop(-1)
        for p in cfg.productions:
            P = p.head
            if p.is_epsilon:
                continue
            if p.body[0] == Q:
                insert(P, a)
    return firstvts


def lastvt(cfg: CFG) -> LastVTs:
    lastvts: LastVTs = dict([(s, set()) for s in cfg.nonterminals])
    stack: List[Tuple[Nonterminal, Terminal]] = []

    def insert(P: Nonterminal, a: Terminal):
        if a not in lastvts[P]:
            lastvts[P].add(a)
            stack.append((P, a))

    for p in cfg.productions:
        P = p.head
        if p.is_epsilon:
            continue
        # P -> a...
        if cfg.is_terminal(p.body[-1]):
            insert(P, p.body[-1])
        # P -> Q a ...
        elif len(p.body) > 1 and cfg.is_nonterminal(
                p.body[-1]) and cfg.is_terminal(p.body[-2]):
            insert(P, p.body[-2])
    while len(stack) != 0:
        Q, a = stack.pop(-1)
        for p in cfg.productions:
            P = p.head
            if p.is_epsilon:
                continue
            if p.body[0] == Q:
                insert(P, a)
    return lastvts


def augmented_grammar(cfg: CFG) -> CFG:
    if Dollar in cfg.terminals:
        return cfg
    # add S' -> S '$' to cfg ?
    terminals = cfg.terminals.copy()
    nonterminals = cfg.nonterminals.copy()
    start_symbol = f'@{cfg.start_symbol}'
    productions = cfg.productions.copy()
    terminals.append(Dollar)
    nonterminals.append(start_symbol)
    productions.append(
        Production(start_symbol, [Dollar, cfg.start_symbol, Dollar]))
    return CFG(terminals, nonterminals, start_symbol, productions)


# 算符优先文法
# 对应的英文是 Operator Precedence Grammar
class OP_Parser:

    def __init__(self, cfg: CFG) -> None:
        # terminals = sorted(cfg.terminals)
        # terminals.append(Dollar)
        # we will calculate operator precedence with the new_cfg,
        # but continue parsing using the original cfg
        new_cfg = augmented_grammar(cfg)
        print('augmented grammar')
        print(new_cfg)
        firstvts = firstvt(new_cfg)
        lastvts = lastvt(new_cfg)
        print('firstvt')
        f = PrettyTable(['', 'FIRSTVT'], align='l')
        f.add_rows([(s, ','.join(firstvts[s])) for s in cfg.nonterminals])
        print(f)
        print('lastvt')
        f = PrettyTable(['', 'LASTVT'], align='l')
        f.add_rows([(s, ','.join(lastvts[s])) for s in cfg.nonterminals])
        print(f)

        self._cfg = cfg

        def insert(a: Terminal, b: Terminal, p: Precedence):
            # we have to check whether table[(a,b)] has different precedence between a,b
            key = Key(a, b)
            if table.get(key) is not None:
                raise RuntimeError(
                    f'failed to insert table[{a},{b}] = {p.value}\nalready got precedence: {table[key].value}'
                )
            table[key] = p

        # construct precedence table
        table: Dict[Key, Precedence] = {}
        for p in new_cfg.productions:
            n = len(p.body)
            for i in range(n - 1):
                x0 = p.body[i]
                x1 = p.body[i + 1]
                if new_cfg.is_terminal(x0) and new_cfg.is_terminal(x1):
                    insert(x0, x1, Precedence.EQUAL)
                elif new_cfg.is_terminal(x0) and new_cfg.is_nonterminal(x1):
                    for a in firstvts[x1]:
                        insert(x0, a, Precedence.LESS)
                elif new_cfg.is_nonterminal(x0) and new_cfg.is_terminal(x1):
                    for a in lastvts[x0]:
                        insert(a, x1, Precedence.GREATER)
                if i < n - 2:
                    x2 = p.body[i + 2]
                    if new_cfg.is_terminal(x0) and new_cfg.is_terminal(
                            x2) and new_cfg.is_nonterminal(x1):
                        insert(x0, x2, Precedence.EQUAL)
        self._table = table
        # we could also construct a precedence function through DAG
        p = PrettyTable(['', *new_cfg.terminals])
        for p0 in new_cfg.terminals:
            row = [str(p0)]
            for p1 in new_cfg.terminals:
                precedence = table.get(Key(p0, p1), Precedence.NONE)
                row.append(precedence.value)
            p.add_row(row)

        print('precedence table:')
        print(p)

        # self._body_map = Dict[Tuple[Union[bool,Symbol]],Production] = []

    def parse(self, _input: List[Terminal]) -> None:
        cfg = self._cfg
        table = self._table
        s: List[Symbol] = [Dollar]

        input = _input.copy()
        input.append(Dollar)

        pos = 0

        def next() -> Symbol:
            nonlocal pos
            curr = input[pos]
            pos += 1
            return curr

        @cache
        def less(a, b) -> bool:
            return table.get(Key(a, b), Precedence.NONE) == Precedence.LESS

        @cache
        def is_terminal(s: Symbol) -> bool:
            return s == Dollar or s in cfg.terminals

        @cache
        def find_production(body: Tuple[Symbol]) -> Optional[Production]:
            for p in cfg.productions:
                if len(p.body) != len(body):
                    continue
                all_match = True
                for i in range(len(body)):
                    a0 = p.body[i]
                    a1 = body[i]
                    if is_terminal(a0) and is_terminal(a1) and a0 == a1:
                        continue
                    if cfg.is_nonterminal(a0) and cfg.is_nonterminal(a1):
                        continue
                    all_match = False
                    break
                if all_match:
                    return p
            return None

        step = 0
        f = PrettyTable(['step', 'stack', 'precedence', 'input', 'action'])
        f.align['stack'] = 'l'
        f.align['input'] = 'r'
        f.align['action'] = 'l'

        def record(precedence: Precedence,
                   action: Action,
                   msg: Optional[Any] = None):
            nonlocal s, step, f, input, pos
            row = [
                f'{step}', ' '.join(s), precedence.value,
                ' '.join(input[pos - 1:]), action.value
            ]
            if msg is not None:
                # accept、shift、reduce, make sure they have the same width 6
                row[-1] = f'{action.value:<6} [ {msg} ]'

            f.add_row(row)
            step += 1

        while True:
            # a must be a terminal
            a = next()
            # j means first terminal in symbol stack
            j = len(s) - 1 if is_terminal(s[-1]) else len(s) - 2
            # 优先级高，说明我们需要归约了
            precedence = table.get(Key(s[j], a), Precedence.NONE)
            # keep reducing
            while precedence == Precedence.GREATER:
                # 找到可以归约的左边界？
                while True:
                    Q = s[j]
                    j -= 1 if is_terminal(s[j - 1]) else 2
                    if less(s[j], Q):
                        break
                body = tuple(s[j + 1:])
                p = find_production(body)
                if p is None:
                    raise RuntimeError(
                        f'fail to find available production with body {body}')
                record(precedence, Action.REDUCE, p)
                s = s[:j + 1]
                s.append(p.head)
                precedence = table.get(Key(s[j], a), Precedence.NONE)
            # shifting
            if precedence == Precedence.LESS or precedence == Precedence.EQUAL:
                if s[-1] == cfg.start_symbol and a == Dollar:
                    record(precedence, Action.ACCEPT)
                    break
                record(precedence, Action.SHIFT, a)
                s.append(a)
            else:
                raise RuntimeError()
        print('pass!!!')
        print(f)


def main():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'example-4.1.txt'), 'r',
              encoding='utf-8') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    print(cfg)
    # new_cfg = eliminate_epsilon(cfg)
    # print(new_cfg)
    parser = OP_Parser(cfg)
    t = list(s for s in 'i*i+i')
    # t = ['i','*','i','+','i',Dollar]
    parser.parse(t)
    # tree = parser.parse(t)
    # g = tree.visualize()
    # g.view(cleanup=True)


if __name__ == '__main__':
    main()