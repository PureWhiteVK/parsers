from functools import cache, cached_property
from typing import Any, Iterable, List, Dict, NamedTuple, Optional, Set, Tuple, NamedTuple, Union, cast
from prettytable import PrettyTable
from enum import Enum, unique

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.cfg import CFG, Production, Symbol, Terminal, Nonterminal, production_map
from src.cfg_utils import first, follow
from src.utils import Dollar, Epsilon, Invalid
from src.cfg_parser import CfgParser

State = int


class TableKey(NamedTuple):
    x: State
    s: Symbol


@unique
class Action(Enum):
    SHIFT = 'shift'
    REDUCE = 'reduce'
    ACCEPT = 'accept'
    GOTO = 'goto'


class TableItem(NamedTuple):
    # action used fo
    form: Action
    value: Union[State, Production]


def augmented_grammar(cfg: CFG) -> CFG:
    count = 0
    for p in cfg.productions:
        if p.head == cfg.start_symbol:
            count += 1
    assert count > 0
    if count == 1:
        return cfg
    # add S' -> S to cfg
    terminals = cfg.terminals.copy()
    nonterminals = cfg.nonterminals.copy()
    start_symbol = f'@{cfg.start_symbol}'
    productions = cfg.productions.copy()
    nonterminals.append(start_symbol)
    productions.append(Production(start_symbol, [cfg.start_symbol]))
    return CFG(terminals, nonterminals, start_symbol, productions)


class Item:

    def __init__(self, production: Production, pos: int) -> None:
        self._production = production
        assert 0 <= pos <= len(production.body)
        self._pos = pos

    @property
    def production(self) -> Production:
        return self._production

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def is_end(self) -> bool:
        return self.pos == len(self.production.body)

    @cached_property
    def after(self) -> Symbol:
        return Epsilon if self.is_end else self.production.body[self.pos]

    def move(self, i: int) -> 'Item':
        return Item(self.production, self.pos + i)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Item):
            return False
        return self.production == __o.production and self.pos == __o.pos

    def __hash__(self) -> int:
        return hash((self.production, self.pos))

    def __repr__(self) -> str:
        body = self.production.body
        pos = self.pos
        head = self.production.head
        s = [head, ' -> ']
        if pos == 0:
            s.append(f'·{"".join(body)}')
        elif pos == len(body):
            s.append(f'{"".join(body)}·')
        else:
            s.append(f'{"".join(body[:pos])}·{"".join(body[pos:])}')
        return ''.join(s)


ItemSet = Tuple[Item, ...]


class SLR_Parser:

    def __init__(self, cfg: CFG) -> None:
        new_cfg = augmented_grammar(cfg)

        terminals = [*cfg.terminals, Dollar]

        self._augmented_cfg = new_cfg
        self._cfg = cfg
        self._terminals = terminals

        productions = new_cfg.productions.copy()
        productions_id = dict([(v, i) for i, v in enumerate(productions)])

        self._productions_id = productions_id
        self._construct_slr_table()

        table = self._table
        items = self._items

        print(new_cfg)

        t = PrettyTable(['state', 'item set'])
        t.align['item set'] = 'l'
        t.add_rows([(i, ' | '.join(map(lambda i: str(i), item)))
                    for i, item in enumerate(items)])
        print(t)

        t = PrettyTable(['pid', 'production'])
        t.align['production'] = 'l'
        t.add_rows([(i, str(p)) for i, p in enumerate(productions)])
        print(t)

        t = PrettyTable(['state', *terminals, *cfg.nonterminals])
        for i in range(len(self._items)):
            row = [f'{i}']
            for a in terminals:
                v = table.get(TableKey(i, a))
                if v is None:
                    row.append('')
                    continue
                if v.form == Action.ACCEPT:
                    row.append('acc')
                elif v.form == Action.REDUCE:
                    row.append(f'r{productions_id[cast(Production,v.value)]}')
                elif v.form == Action.SHIFT:
                    row.append(f's{v.value}')
            for a in cfg.nonterminals:
                v = table.get(TableKey(i, a))
                if v is None:
                    row.append('')
                    continue
                row.append(f'{v.value}')
            t.add_row(row)
        print(t)

    def _construct_slr_table(self):
        cfg = self._augmented_cfg
        p_map = production_map(cfg.productions)
        start_prod = p_map[cfg.start_symbol][0]

        @cache
        def closure(I: Tuple[Item, ...]) -> ItemSet:
            queue = list(I)
            res = set(I)
            while len(queue) != 0:
                curr = queue.pop(0)
                if cfg.is_terminal(curr.after):
                    continue
                for p in p_map.get(curr.after, []):
                    item = Item(p, 0)
                    if item not in res:
                        res.add(item)
                        queue.append(item)
            # the order matters, so we can't use set
            return tuple(
                sorted(res,
                       key=lambda i:
                       (self._productions_id[i.production], i.pos)))

        @cache
        def goto(I: ItemSet, X: Symbol) -> ItemSet:
            return closure(tuple([i.move(1) for i in I if i.after == X]))

        initial_set = closure((Item(start_prod, 0), ))
        item_sets: Set[ItemSet] = {initial_set}
        items: List[ItemSet] = [initial_set]
        items_id: Dict[ItemSet, int] = {initial_set: 0}

        table: Dict[TableKey, TableItem] = {}
        for I in items:
            for X in cfg.symbols:
                c = goto(I, X)
                if len(c) == 0:
                    continue
                # create new item_set
                if c not in item_sets:
                    item_sets.add(c)
                    items_id[c] = len(items)
                    items.append(c)

        firsts = first(cfg)
        follows = follow(cfg, firsts)

        for Ii in items:
            Ii_id = items_id[Ii]
            for i in Ii:
                if cfg.is_terminal(i.after):
                    a = i.after
                    Ij = goto(Ii, a)
                    if len(Ij) == 0:
                        continue
                    Ij_id = items_id[Ij]
                    table[TableKey(Ii_id, a)] = TableItem(Action.SHIFT, Ij_id)
                if i.is_end:
                    A = i.production.head
                    if A != cfg.start_symbol:
                        # reduce
                        for a in follows[A]:
                            table[TableKey(Ii_id, a)] = TableItem(
                                Action.REDUCE, i.production)
                    else:
                        table[TableKey(Ii_id, Dollar)] = TableItem(
                            Action.ACCEPT, i.production)
            for A in cfg.nonterminals:
                Ij = goto(Ii, A)
                if len(Ij) != 0:
                    Ij_id = items_id[Ij]
                    table[TableKey(Ii_id, A)] = TableItem(Action.GOTO, Ij_id)

        self._table = table
        self._items = items
        self._items_id = items_id

    def parse(self, _input: List[Symbol]) -> None:
        cfg = self._augmented_cfg
        table = self._table
        stack: List[State] = [0]
        symbols: List[Symbol] = []

        input = _input.copy()
        input.append(Dollar)

        pos = 0

        def next() -> Symbol:
            nonlocal pos
            curr = input[pos]
            pos += 1
            return curr
        
        step = 0
        f = PrettyTable(['step', 'stack', 'symbols', 'input', 'action'])
        f.align['stack'] = 'l'
        f.align['input'] = 'r'
        f.align['symbols'] = 'l'
        f.align['action'] = 'l'

        def record(action: Action,
                   msg: Optional[Any] = None):
            nonlocal stack, step, f, input, pos, symbols
            row = [
                f'{step}', ' '.join(map(lambda s:str(s),stack)), ' '.join(symbols),
                ' '.join(input[pos - 1:]), action.value
            ]
            if msg is not None:
                # accept、shift、reduce, make sure they have the same width 6
                row[-1] = f'{action.value:<6} [ {msg} ]'

            f.add_row(row)
            step += 1

        a = next()
        while True:
            state = stack[-1]
            action = table[TableKey(state, a)]
            assert action
            if action.form == Action.SHIFT:
                next_state = cast(State, action.value)
                # print(f'shift: {a}')
                record(action.form,a)
                symbols.append(a)
                stack.append(next_state)
                a = next()
            elif action.form == Action.REDUCE:
                prod = cast(Production, action.value)
                # print(f'reduce: {prod}')
                record(action.form,prod)
                A = prod.head
                for _ in range(len(prod.body)):
                    stack.pop()
                    symbols.pop()
                t = stack[-1]
                goto = table[TableKey(t, A)]
                assert goto
                next_state = cast(State, goto.value)
                stack.append(next_state)
                symbols.append(A)
            elif action.form == Action.ACCEPT:
                record(action.form)
                break
        print('pass!!!')
        print(f)


def main():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'test.txt'), 'r',
              encoding='utf-8') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    parser = SLR_Parser(cfg)
    t = list(s for s in 'babb')
    # t = ['i','*','i','+','i',Dollar]
    parser.parse(t)
    # tree = parser.parse(t)
    # g = tree.visualize()
    # g.view(cleanup=True)


if __name__ == '__main__':
    main()