from typing import List, Dict, NamedTuple, Optional
from prettytable import PrettyTable

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.cfg import CFG, Symbol, Terminal, Nonterminal, Production
from src.cfg_utils import first, follow, first_symbols
from src.utils import Dollar, Epsilon
from src.parse_tree import ParseTree
from src.cfg_parser import CfgParser


class Key(NamedTuple):
    symbol: Nonterminal
    input: Terminal


class LL1_Parser:

    def __init__(self, cfg: CFG) -> None:
        print(cfg)
        firsts = first(cfg)
        follows = follow(cfg, firsts)
        print('first sets')
        f = PrettyTable(['', 'FIRST'], align='l')
        f.add_rows([(s, ','.join(firsts[s])) for s in cfg.nonterminals])
        print(f)
        print('follow sets')
        f = PrettyTable(['', 'FOLLOW'], align='l')
        f.add_rows([(s, ','.join(follows[s])) for s in cfg.nonterminals])
        print(f)
        table: Dict[Key, Production] = {}
        for p in cfg.productions:
            A = p.head
            first_alpha = first_symbols(tuple(p.body), firsts)
            for a in first_alpha:
                table[Key(A, a)] = p
            if Epsilon in first_alpha:
                for b in follows[A]:
                    table[Key(A, b)] = p

        terminals = [*cfg.terminals, Dollar]
        if Epsilon in terminals:
            terminals.remove(Epsilon)

        t = PrettyTable(['', *terminals])

        for s in cfg.nonterminals:
            row = [s]
            for n in terminals:
                res = table.get(Key(s, n), None)
                row.append('' if res is None else str(res))
            t.add_row(row)
        print('parse table:')
        print(t)
        self._M = table
        self._cfg = cfg

    def parse(self, _input: List[Terminal]) -> ParseTree:
        input = _input.copy()
        input.append(Dollar)

        pos = 0

        def next() -> Symbol:
            nonlocal pos
            curr = input[pos]
            pos += 1
            return curr

        root = ParseTree.Node(self._cfg.start_symbol)
        node_stack: List[ParseTree.Node] = [root]
        stack: List[Symbol] = [Dollar, self._cfg.start_symbol]
        t = PrettyTable(['step', 'stack', 'input', 'production'])
        t.align['stack'] = 'l'
        t.align['input'] = 'r'
        t.align['production'] = 'l'
        n_iter = 0
        a = next()
        while True:
            row = [str(n_iter), ' '.join(stack), ' '.join(input[pos - 1:]), '']
            x = stack.pop(-1)
            if self._cfg.is_terminal(x):
                if x == a:
                    a = next()
                    node_stack.pop()
                else:
                    raise RuntimeError(str(row))
            elif x == Dollar:
                if x == a:
                    break
                else:
                    raise RuntimeError(str(row))
            elif self._M.get(Key(x, a)) is not None:
                p = self._M[Key(x, a)]
                node = node_stack.pop(-1)
                row[3] = str(p)
                if not p.is_epsilon:
                    children = list(map(lambda s: ParseTree.Node(s), p.body))
                    node.children = children
                    node_stack.extend(children[::-1])
                    stack.extend(p.body[::-1])
                else:
                    node.children = [ParseTree.Node(Epsilon)]
            else:
                raise RuntimeError(str(row))
            t.add_row(row)
            n_iter += 1
        print('pass!!')
        print(t)
        return ParseTree(root)


def main():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'example-4.2.txt'), 'r') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    # new_cfg = eliminate_epsilon(cfg)
    # print(new_cfg)
    parser = LL1_Parser(cfg)
    t = list(s for s in 'i*i+i')
    tree = parser.parse(t)
    # g = tree.visualize()
    # g.view(cleanup=True)


if __name__ == '__main__':
    main()
