from copy import deepcopy
from functools import cache
from typing import List, Dict, Tuple, Set, Optional, Sized, cast
from typing_extensions import TypeAlias

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.cfg import Production, CFG, Trie, Symbol, Nonterminal, Terminal, production_map
from src.parse_tree import ParseTree
from src.cfg_parser import CfgParser
from src.utils import Dollar, Epsilon, OUTPUT_DIR, NameAllocator


def left_factoring(cfg: CFG) -> CFG:
    productions = set(cfg.productions)
    terminals = set(cfg.terminals)
    nonterminals = set(cfg.nonterminals)

    # 使用 trie 来找出公共前缀
    change = True
    while change:
        change = False
        p_map = production_map(productions)
        for s, s_prods in p_map.items():
            # 找出公共前缀
            trie = Trie()
            for prod in s_prods:
                trie.add(prod)
            # g = trie.visualize()
            # g.view(cleanup=True)
            # 此处我们需要计算的是存在的最长公共前缀？
            prefix, prods = trie.longest_common_prefix()
            if len(prefix) == 0:
                break
            new_s = Nonterminal(f'{s}\'')
            nonterminals.add(new_s)
            change = True
            productions.difference_update(prods)
            productions.add(Production(s, [*prefix, new_s]))
            for p in prods:
                new_p = Production(new_s, p.body[len(prefix):].copy())
                productions.add(new_p)
                if new_p.is_epsilon:
                    terminals.add(Epsilon)
    return CFG(terminals, nonterminals, cfg.start_symbol, productions)


def eliminate_left_recursion(cfg: CFG,
                             order: Optional[List[Nonterminal]] = None) -> CFG:
    nonterminals = list(cfg.nonterminals)
    if order is not None:
        nonterminals = order
    print('V order:', nonterminals)
    terminals = list(cfg.terminals)
    p_map = production_map(cfg.productions)
    n = len(nonterminals)

    def eliminate_direct(nt: Nonterminal):
        r_p: List[List[Symbol]] = []
        n_p: List[List[Symbol]] = []
        # A -> A a_1 | A a_2 ... | A a_m | b_1 | b_2 | ... | b_n
        # A -> b_1 | b_2 | ... | b_n | b_1 A' | b_2 A' | ... | b_n A'
        # A' -> Epsilon | a_1 A' | a_2 A' | ... | a_m A'
        for prods in p_map.get(nt, []):
            b = prods.body
            # left recursion production
            if len(b) > 0 and b[0] == nt:
                r_p.append(b)
            else:
                # non-recursion production
                n_p.append(b)
        if len(r_p) == 0:
            # no need to eliminate
            return
        new_nt = Nonterminal(f'{nt}\'')
        nonterminals.append(new_nt)
        new_bodys: List[List[Symbol]] = [[*b, new_nt] for b in n_p]
        new_bodys_nt: List[List[Symbol]] = [[*b[1:], new_nt] for b in r_p]
        # new_nt -> epsilon
        new_bodys_nt.append([])
        terminals.append(Epsilon)
        p_map[nt] = [Production(nt, b) for b in new_bodys]
        p_map[new_nt] = [Production(new_nt, b) for b in new_bodys_nt]

    def substitute(Vi: Nonterminal, Vj: Nonterminal):
        vj_prods: List[Production] = p_map.get(Vj, [])
        if len(vj_prods) == 0:
            return
        new_vi_prods: List[Production] = []
        for prod in p_map.get(Vi, []):
            b = prod.body
            if len(b) > 0 and b[0] == Vj:
                for vj_prod in vj_prods:
                    new_vi_prods.append(Production(Vi,
                                                   [*vj_prod.body, *b[1:]]))
                continue
            new_vi_prods.append(prod)
        p_map[Vi] = new_vi_prods

    for i in range(n):
        Vi = nonterminals[i]
        for j in range(0, i):
            Vj = nonterminals[j]
            substitute(Vi, Vj)
        eliminate_direct(Vi)

    productions: Set[Production] = set()
    for prod in p_map.values():
        productions.update(prod)

    return CFG(set(terminals), set(nonterminals), cfg.start_symbol,
               productions)


def eliminate_useless(cfg: CFG) -> CFG:
    generating: Set[Symbol] = set(cfg.terminals)
    # start from terminal sets, loop through all productions to determine whether the right part is generative
    changed = True
    while changed:
        changed = False
        # o(n^2) algorithm
        for p in cfg.productions:
            if p.head in generating:
                continue
            if p.is_epsilon or all(s in generating for s in p.body):
                generating.add(p.head)
                changed = True
    print(f'generating symbols: {generating}')
    # 消除所有的non-generative production
    new_productions: Set[Production] = set(
        p for p in cfg.productions
        if p.head in generating and all(s in generating for s in p.body))

    reachable: Set[Symbol] = set()
    queue: List[Symbol] = list([cfg.start_symbol])
    p_map = production_map(new_productions)
    # o(n^2) ?
    while len(queue) != 0:
        curr = queue.pop(0)
        if curr in reachable:
            continue
        reachable.add(curr)
        if cfg.is_terminal(curr):
            continue
        for prod in p_map.get(cast(Nonterminal, curr), []):
            queue.extend(prod.body)
    print(f'reachable symbols: {reachable}')
    new_productions = set(p for p in cfg.productions
                          if p.head in reachable and all(s in reachable
                                                         for s in p.body))
    new_terminals = set(t for t in cfg.terminals if t in reachable)
    new_nonterminals = set(n for n in cfg.nonterminals if n in reachable)
    return CFG(new_terminals, new_nonterminals, cfg.start_symbol,
               new_productions)


def eliminate_epsilon(cfg: CFG) -> CFG:
    if not Epsilon in cfg.terminals:
        print('no need to eliminate epsilon')
        return cfg
    # find nullable nonterminals
    nullable: Set[Nonterminal] = set()
    change = True
    while change:
        change = False
        for p in cfg.productions:
            if p.head in nullable:
                continue
            if p.is_epsilon or all(s in nullable for s in p.body):
                nullable.add(p.head)
                change = True
    print(f'nullable symbols: {nullable}')
    if cfg.start_symbol in nullable:
        print(
            f'L(G) contains {Epsilon}, {eliminate_epsilon.__name__} will generate L(H) = L(G) - {{{Epsilon}}}'
        )

    def gen_subsets(subset: List[int], positions: List[int],
                    body: List[Symbol], used: List[bool], index: int,
                    res: List[List[int]]) -> None:
        if index > len(positions):
            return
        res.append(deepcopy(subset))
        for i in range(index, len(positions)):
            if i > 0 and used[i - 1] == False and \
               body[positions[i - 1]] == body[positions[i]]:
                continue
            subset.append(positions[i])
            used[i] = True
            gen_subsets(subset, positions, body, used, i + 1, res)
            subset.pop(-1)
            used[i] = False

    new_productions: Set[Production] = set()
    new_terminals = deepcopy(cfg.terminals)
    new_terminals.remove(Epsilon)

    for p in cfg.productions:
        if p.is_epsilon:
            continue
        nullable_index: List[int] = list(i for i, s in enumerate(p.body)
                                         if s in nullable)
        normal_index: List[int] = list(i for i, s in enumerate(p.body)
                                       if s not in nullable)
        # eliminate nullable symbols
        subsets: List[List[int]] = []
        gen_subsets([], nullable_index, p.body,
                    [False for _ in range(len(nullable_index))], 0, subsets)
        for subset in subsets:
            subset.extend(normal_index)
            subset.sort()
            new_body = [p.body[i] for i in subset]
            if not len(new_body) == 0:
                new_productions.add(Production(p.head, new_body))
    print(new_terminals)
    print(cfg.nonterminals)
    print(cfg.start_symbol)
    print(new_productions)
    return CFG(new_terminals, cfg.nonterminals, cfg.start_symbol,
               new_productions)


def eliminate_unit(cfg: CFG) -> CFG:
    # get unit pair
    unit_pairs: Set[Tuple[Nonterminal, Nonterminal]] = set()
    stack: List[Tuple[Nonterminal,
                      Nonterminal]] = [(v, v) for v in cfg.nonterminals]
    p_map = production_map(cfg.productions)

    while len(stack) != 0:
        u = stack.pop(-1)
        unit_pairs.add(u)
        for prods in p_map.get(u[1], []):
            body = prods.body
            if len(body) == 1 and cfg.is_nonterminal(body[0]):
                candidate = (u[0], cast(Nonterminal, body[0]))
                if candidate not in unit_pairs:
                    stack.append(candidate)

    print(f'unit pairs: {unit_pairs}')
    # remove all unit production
    productions: Set[Production] = set(p for p in cfg.productions if (
        len(p.body) == 1 and cfg.is_terminal(p.body[0])) or len(p.body) != 1)
    p_map = production_map(productions)
    # add productions
    changed = True
    # o(n^2) algorithm
    while changed:
        changed = False
        for v0, v1 in unit_pairs:
            for prod in p_map.get(v1, []):
                p = Production(v0, prod.body)
                if p not in productions:
                    productions.add(p)
                    p_map[v1].append(Production(v1, prod.body))
                    changed = True

    return CFG(cfg.terminals, cfg.nonterminals, cfg.start_symbol, productions)


def cfg_to_cnf(cfg: CFG) -> CFG:

    name_allocator = NameAllocator()
    # we need to make an S' -> S, S' -> ε
    # has_epsilon = cfg.has_epsilon
    new_cfg = eliminate_epsilon(cfg)
    new_cfg = eliminate_unit(new_cfg)
    new_cfg = eliminate_useless(new_cfg)

    productions = deepcopy(new_cfg.productions)
    # terminals = deepcopy(cfg.terminals)
    nonterminals = set(new_cfg.nonterminals)

    terminal_map: Dict[Terminal, Nonterminal] = {}
    for p in new_cfg.productions:
        if len(p.body) == 1 and new_cfg.is_terminal(p.body[0]):
            t = cast(Terminal, p.body[0])
            terminal_map[t] = p.head
    for t in new_cfg.terminals:
        if not terminal_map.get(t, False):
            n = Nonterminal(name_allocator.next)
            nonterminals.add(n)
            terminal_map[t] = n
            productions.append(Production(n, [t]))
    for p in productions:
        if len(p.body) > 1:
            for i, s in enumerate(p.body):
                if new_cfg.is_terminal(s):
                    p.body[i] = terminal_map[cast(Terminal, s)]
    queue: List[Production] = productions.copy()
    productions.clear()
    while len(queue) != 0:
        curr = queue.pop(0)
        if len(curr.body) > 2:
            n = Nonterminal(name_allocator.next)
            nonterminals.add(n)
            queue.append(Production(n, curr.body[1:]))
            queue.append(Production(curr.head, [curr.body[0], n]))
            continue
        productions.append(curr)
    terminals = new_cfg.terminals
    start_symbol = new_cfg.start_symbol
    # augment grammar
    # if has_epsilon:
    #     terminals = deepcopy(terminals)
    #     start_symbol = Nonterminal(name_allocator.next_name)
    #     terminals.add(Epsilon)
    #     nonterminals.add(start_symbol)
    #     productions.extend([
    #         Production(start_symbol, [new_cfg.start_symbol]),
    #         Production(start_symbol)
    #     ])
    return CFG(terminals, nonterminals, start_symbol, productions)


def is_cnf(cfg: CFG) -> bool:
    # no useless symbols
    # A -> BC or A -> a
    for p in cfg.productions:
        if len(p.body) == 1 and cfg.is_terminal(p.body[0]):
            continue
        if len(p.body) == 2 and cfg.is_nonterminal(
                p.body[0]) and cfg.is_nonterminal(p.body[1]):
            continue
        return False
    return True


# 可以递归的输出所有的可能的语法分析树，但是时间复杂度很高（？）
def cyk(cnf: CFG, w: List[Terminal]) -> Optional[ParseTree]:
    n = len(w)
    dp: List[List[Set[Nonterminal]]] = [[set() for _ in range(n)]
                                        for _ in range(n)]
    # dp[i][j] 表示从 w[i] 开始到 w[j] 的那一段字符串所对应的非终结符
    # dp[i][i] = if has production A -> a and w[i] == a
    path: List[List[Dict[int,
                         List[Production]]]] = [[dict() for _ in range(n)]
                                                for _ in range(n)]

    body_map: Dict[Tuple[Symbol, ...], List[Nonterminal]] = {}
    for p in cnf.productions:
        key = tuple(p.body)
        l = body_map.get(key, [])
        l.append(p.head)
        body_map[key] = l

    def print_table(t: List[List[Sized]]):
        from prettytable import PrettyTable
        p_table = PrettyTable(header=False)
        p_table.add_row(['', *w])
        p_table.add_rows(
            [[w[r], *map(lambda s: '-' if len(s) == 0 else s, t[r])]
             for r in range(n)])
        print(p_table)

    # 递归构造解析数（只返回一种可行的，有可能存在多个，实际上可以找到所有的）
    def construct_tree(i: int, j: int, nt: Nonterminal) -> ParseTree.Node:
        node = ParseTree.Node(nt)
        if i == j:
            node.children = [ParseTree.Node(w[i])]
        else:
            prev = path[i][j]
            found = False
            for k, v in prev.items():
                # find one possible derivation
                for p in v:
                    if p.head == nt:
                        left = construct_tree(i, k,
                                              cast(Nonterminal, p.body[0]))
                        right = construct_tree(k + 1, j,
                                               cast(Nonterminal, p.body[1]))
                        node.children = [left, right]
                        found = True
                        break
                if found:
                    break
        return node

    # dynamic programing
    for i in range(n):
        key = body_map.get((w[i], ), [])
        dp[i][i].update(key)
        path[i][i][i] = list(Production(h, [w[i]]) for h in key)
    for l in range(1, n):
        for s in range(0, n - l):
            e = s + l
            curr_p: Dict[int, List[Production]] = path[s][e]
            for k in range(s, e):
                c0 = dp[s][k]
                c1 = dp[k + 1][e]
                res_k = curr_p.get(k, [])
                curr_p[k] = res_k
                # check pair
                # 笛卡尔积
                candidates = [(v0, v1) for v0 in c0 for v1 in c1]
                for key in candidates:
                    res = body_map.get(key, None)
                    if res is not None:
                        dp[s][e].update(res)
                        res_k.extend(Production(h, list(key)) for h in res)
    if cnf.start_symbol in dp[0][n - 1]:
        return ParseTree(construct_tree(0, n - 1, cnf.start_symbol))
    return None


FirstSets: TypeAlias = Dict[Symbol, Set[Nonterminal]]
FollowSets: TypeAlias = Dict[Symbol, Set[Nonterminal]]


def first(cfg: CFG) -> FirstSets:
    firsts: FirstSets = {}
    for s in cfg.terminals:
        firsts[s] = {s}
    for s in cfg.nonterminals:
        firsts[s] = set()

    productions: List[Production] = []

    for p in cfg.productions:
        first_p = firsts[p.head]
        if p.is_epsilon:
            first_p.add(Epsilon)
            continue
        if cfg.is_terminal(p.body[0]):
            first_p.add(p.body[0])
            continue
        productions.append(p)

    def update(first: Set[Symbol], candidate: Set[Symbol]) -> bool:
        updated = False
        for v in candidate:
            if v not in first:
                updated = True
                first.add(v)
        return updated

    change = True
    while change:
        change = False
        for p in productions:
            first_p = firsts[p.head]
            for s in p.body:
                first_s = firsts[s]
                change = update(first_p, first_s)
                if Epsilon not in first_s:
                    break
    return firsts


def first_symbols(symbols: Tuple[Symbol],
                  first_sets: FirstSets) -> Set[Nonterminal]:
    firsts: Set[Symbol] = set()
    all_epsilon = True
    for X in symbols:
        if not all_epsilon:
            break
        first_X = first_sets[X]
        all_epsilon = False
        for a in first_X:
            if a == Epsilon:
                all_epsilon = True
            else:
                firsts.add(a)
    if all_epsilon:
        firsts.add(Epsilon)
    return firsts


def follow(cfg: CFG, first_sets: FirstSets) -> FollowSets:
    follows: FollowSets = dict([(s, set()) for s in cfg.nonterminals])
    # follow(S) 至少包含一个 '$'
    follows[cfg.start_symbol].add(Dollar)

    def update(follow: Set[Nonterminal], candidate: Set[Nonterminal]) -> bool:
        updated = False
        for v in candidate:
            if v == Epsilon:
                continue
            if v not in follow:
                updated = True
                follow.add(v)
        return updated

    change = True
    while change:
        change = False
        for p in cfg.productions:
            follow_p = follows[p.head]
            for i, s in enumerate(p.body):
                if cfg.is_terminal(s):
                    continue
                follow_s = follows[s]
                B = tuple() if i == len(p.body) - 1 else tuple(p.body[i + 1:])
                first_B = first_symbols(B, first_sets)
                change = update(follow_s, first_B)
                if Epsilon in first_B:
                    change = update(follow_s, follow_p)
    return follows


def test_left_recursion():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'example-4.3.txt'), 'r') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    print(f'before:\n{cfg}')
    new_cfg = eliminate_left_recursion(cfg)
    print(f'after:\n{new_cfg}')


def test_left_factoring():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'example-4.2.txt'), 'r') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    print(f'before:\n{cfg}')
    new_cfg = eliminate_left_recursion(cfg, ['E', 'T', 'F'])
    print(f'after:\n{new_cfg}')


def test_first_follow():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'example-4.2.txt'), 'r') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    print(f'before:\n{cfg}')
    new_cfg = eliminate_left_recursion(cfg, ['E', 'T', 'F'])
    print(f'after:\n{new_cfg}')
    firsts = first(new_cfg)
    from prettytable import PrettyTable
    f = PrettyTable(['V', 'FIRST'])
    f.add_rows([(s, firsts[s]) for s in new_cfg.nonterminals])
    print(f)
    follows = follow(new_cfg, firsts)
    f = PrettyTable(['V', 'FOLLOW'])
    f.add_rows([(s, follows[s]) for s in new_cfg.nonterminals])
    print(f)


def test_cyk():
    cfg: Optional[CFG] = None
    from utils import ROOT_DIR
    with open(os.path.join(ROOT_DIR, 'example-4.2.txt'), 'r',
              encoding='utf-8') as f:
        cfg = CfgParser().parse(f.read())
    if cfg is None:
        exit(-1)
    print(cfg)
    # new_cfg = eliminate_left_recursion(cfg)
    # print(new_cfg)
    new_cfg = cfg_to_cnf(cfg)
    print(new_cfg)
    test_str = '(i+i)*i'
    tree = cyk(new_cfg, [s for s in test_str])
    if tree is not None:
        g = tree.visualize()
        g.view(directory=OUTPUT_DIR, cleanup=True)
    else:
        print(f'failed to parse {test_str}')


def main():
    test_cyk()
    # cfg = CFG(terminals, nonterminals, start_symbol, productions)
    # print(cfg)
    # # new_cfg = eliminate_left_recursion(cfg)
    # # print(new_cfg)
    # new_cfg = cfg_to_cnf(cfg)
    # print(new_cfg)
    # test_str = 'a+a*a'
    # tree = cyk(new_cfg, [s for s in test_str])
    # if tree is not None:
    #     g = tree.visualize()
    #     g.view(directory=OUTPUT_DIR, cleanup=True)
    # else:
    #     print(f'failed to parse {test_str}')

    # from prettytable import PrettyTable
    # first_sets = first(cfg)
    # f = PrettyTable(['V', 'FIRST'])
    # f.add_rows([s, first_sets[s]] for s in cfg.nonterminals)
    # print(f)
    # follow_sets = follow(cfg, first_sets)
    # f = PrettyTable(['V', 'FOLLOW'])
    # f.add_rows([s, follow_sets[s]] for s in cfg.nonterminals)
    # print(f)


if __name__ == '__main__':
    main()