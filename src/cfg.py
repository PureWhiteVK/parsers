from functools import cache, cached_property
import json
from typing import Iterable, List, Dict, Any, Tuple, Union, Optional, Set, cast
from typing_extensions import TypeAlias
import graphviz

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import Epsilon, check_type, check_array_type

Terminal: TypeAlias = str
Nonterminal: TypeAlias = str
Symbol = Union[Terminal, Nonterminal]


class Production:

    def __init__(self,
                 head: Nonterminal,
                 body: Optional[List[Symbol]] = None) -> None:
        self._head = head
        self._body = []

        if body is not None:
            if len(body) == 1 and body[0] == Epsilon:
                return
            check_array_type(body,
                             Symbol,
                             list,
                             'Production.body',
                             allow_empty=True)
            self._body = body

    @property
    def head(self) -> Nonterminal:
        return self._head

    @property
    def body(self) -> List[Symbol]:
        return self._body

    @property
    def is_epsilon(self) -> bool:
        return len(self._body) == 0

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Production):
            return False
        return self.head == __o.head and self.body == __o.body

    def __hash__(self) -> int:
        return hash(tuple([self._head, *self._body]))

    @staticmethod
    def from_dict(data: Dict) -> 'Production':
        head = cast(Nonterminal, data.get('head'))
        body = cast(List[Symbol], data.get('body'))
        check_type(head, Nonterminal, 'Production.head')
        check_array_type(body, Symbol, list, 'Production.body', False)
        return Production(head, body)

    def to_dict(self) -> Dict[str, Any]:
        return {'head': self._head, 'body': self._body, 'type': 'Production'}

    def __repr__(self) -> str:
        return f'{self._head} -> {Epsilon if self.is_epsilon else " ".join(self._body)}'


def production_map(
        productions: Iterable[Production]
) -> Dict[Nonterminal, List[Production]]:
    res: Dict[Nonterminal, List[Production]] = {}
    for p in productions:
        l = res.get(p.head, list())
        l.append(p)
        res[p.head] = l
    return res


class CFG:

    def __init__(self, terminals: Iterable[Terminal],
                 nonterminals: Iterable[Nonterminal],
                 start_symbol: Nonterminal,
                 productions: Iterable[Production]) -> None:
        self._terminals = sorted(terminals)
        self._nonterminals = sorted(nonterminals)
        self._start_symbol = start_symbol
        assert start_symbol in nonterminals, f'Start symbol {start_symbol} must be nonterminal!'
        self._productions = sorted(productions, key=lambda p: (p.head,*p.body))
        for p in productions:
            assert p.head in nonterminals, f'Production.head {p.head} must be nonterminal!'
            for s in p.body:
                assert s in nonterminals or s in terminals, f'Production.body {s} must be terminal or nonterminal!'

    def __repr__(self) -> str:
        res = ''
        # res += f'terminals: {self._terminals}\nnonterminals: {self._nonterminals}\nstart symbol: {self._start_symbol}\nproductions:\n'
        max_name_length = len(max(self._nonterminals, key=lambda a: len(a)))

        def pad(s: Symbol) -> str:
            return s + ' ' * (max_name_length - len(s))

        def format_body(body: List[Symbol]) -> str:
            if len(body) == 0:
                return Epsilon
            res = []
            for s in body:
                if self.is_terminal(s):
                    res.append(f'{s!r}')
                else:
                    res.append(s)
            return ' '.join(res)

        def pretty_format(symbol: Nonterminal) -> str:
            s = ''
            start = True
            p_map = production_map(self._productions)
            for b in p_map.get(symbol, []):
                if start:
                    s += f'{pad(symbol)} : {format_body(b.body)}'
                    start = False
                    continue
                s += f'\n{" " * max_name_length} | {format_body(b.body)}'
            s += f'\n{" " * max_name_length} ;'
            return s

        res += pretty_format(self._start_symbol)
        for s in self._nonterminals:
            if s == self._start_symbol:
                continue
            res += f'\n{pretty_format(s)}'
        return res
    # -4646677634834017184
    def to_file(self, filename) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            # f.encoding = 'utf-8'
            f.write(str(self))

    @property
    def terminals(self) -> List[Terminal]:
        return self._terminals

    @property
    def nonterminals(self) -> List[Nonterminal]:
        return self._nonterminals

    @property
    def start_symbol(self) -> Nonterminal:
        return self._start_symbol

    @property
    def productions(self) -> List[Production]:
        return self._productions

    @cached_property
    def symbols(self) -> List[Symbol]:
        return list([*self.nonterminals, *self.terminals])

    @cache
    def is_nonterminal(self, obj: Symbol) -> bool:
        return obj in self._nonterminals

    @cache
    def is_terminal(self, obj: Symbol) -> bool:
        return obj in self._terminals

    @staticmethod
    def from_dict(data: Dict) -> Union['CFG', 'Production']:
        type_name = cast(str, data.get('type'))
        if type_name == 'Production':
            return Production.from_dict(data)
        elif type_name != 'CFG':
            raise ValueError(
                f'type field of json object must be `Production` or `CFG`, requested: {type_name}'
            )
        terminals = cast(List[Terminal], data.get('terminals'))
        nonterminals = cast(List[Nonterminal], data.get('nonterminals'))
        productions = cast(List[Production], data.get('productions'))
        start_symbol = cast(Nonterminal, data.get('start_symbol'))
        check_type(start_symbol, Nonterminal, 'start_symbol')
        check_array_type(terminals, Terminal, list, 'terminals')
        check_array_type(nonterminals, Nonterminal, list, 'nonterminals')
        check_array_type(productions, Production, list, 'productions')
        return CFG(terminals, nonterminals, start_symbol, productions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'CFG',
            'terminals': list(self._terminals),
            'nonterminals': list(self._nonterminals),
            'start_symbol': self._start_symbol,
            'productions': list(p.to_dict() for p in self._productions)
        }


class Trie:

    class Node:

        def __init__(self, symbol: Symbol) -> None:
            self.curr: Symbol = symbol
            self.count: int = 0
            self.production: Optional[Production] = None
            self.next: Dict[Symbol, 'Trie.Node'] = {}

        def __repr__(self) -> str:
            return f'<curr={self.curr},count={self.count}>'

    def __init__(self) -> None:
        # virtual node
        self.root: 'Trie.Node' = Trie.Node(Nonterminal('Head'))

    def add(self, production: Production) -> None:
        curr_node = self.root
        curr_node.count += 1
        for b in production.body:
            next_node = curr_node.next.get(b, Trie.Node(b))
            next_node.count += 1
            curr_node.next[b] = next_node
            curr_node = next_node
        curr_node.production = production

    def visualize(self) -> graphviz.Digraph:
        g = graphviz.Digraph('trie')
        node_id = 0

        def recursive(node: 'Trie.Node', parent_id: int = -1) -> None:
            nonlocal node_id
            current_id = node_id
            if node.production is not None:
                # underscore
                g.node(f'n{current_id}',
                       label=f'<<U>{node.curr}</U>>',
                       shape='none')
            else:
                g.node(f'n{current_id}', label=node.curr, shape='none')
            node_id += 1
            if parent_id != -1:
                g.edge(f'n{parent_id}', f'n{current_id}', dir='none')
            for child in node.next.values():
                recursive(child, current_id)

        recursive(self.root)
        return g

    # 返回 prefix 以及所有包含该前缀的字符
    def longest_common_prefix(self) -> Tuple[List[Symbol], List[Production]]:
        # 回溯找到最长的公共前缀
        def recursive(root: 'Trie.Node') -> List[Symbol]:
            if root.count < 2:
                # no shared
                return []
            res: List[Symbol] = []
            for c in root.next.values():
                curr = recursive(c)
                if len(curr) > len(res):
                    res = curr
            if root != self.root:
                res.insert(0, root.curr)
            return res

        prefix = recursive(self.root)
        if len(prefix) == 0:
            return [], []
        # 返回该前缀下的所有可能的Production
        res: List[Production] = []
        curr = self.root
        for p in prefix:
            curr = curr.next[p]
        # 找到该前缀下的所有可能产生式
        queue: List['Trie.Node'] = [curr]
        while len(queue) != 0:
            curr = queue.pop(0)
            if curr.production is not None:
                res.append(curr.production)
            for c in curr.next.values():
                queue.append(c)
        return prefix, res


def main():
    """
    E → E + T | T
    T → T * F | F
    F → ( E ) | a
    """
    terminals: Set[Terminal] = set(['+', '*', '(', ')', 'a'])
    nonterminals: Set[Nonterminal] = set(['E', 'T', 'F'])
    start_symbol: Nonterminal = 'E'

    def p(*args):
        head = cast(Nonterminal, args[0])
        body = cast(List[Symbol], list(args[1:]))
        return Production(head, body)

    productions: Set[Production] = {
        p('E', 'E', '+', 'T'),
        p('E', 'T'),
        p('T', 'T', '*', 'F'),
        p('T', 'F'),
        p('F', '(', 'E', ')'),
        p('F', 'a'),
    }

    cfg = CFG(terminals, nonterminals, start_symbol, productions)

    s = json.dumps(cfg.to_dict())
    new_cfg = json.loads(s, object_hook=CFG.from_dict)

    print(new_cfg)

    from utils import OUTPUT_DIR
    from os.path import join

    with open(join(OUTPUT_DIR, 'test_cfg.json'), 'w') as f:
        json.dump(new_cfg.to_dict(), f)


if __name__ == '__main__':
    main()
