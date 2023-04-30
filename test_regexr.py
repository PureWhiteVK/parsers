from __future__ import annotations
import lark
import json5
from dataclasses import dataclass
from src.parse_tree import ParseTree
from src.regexr import alphabet_exr_nfa, epsilon_exr_nfa, union, concat, closure, Epsilon, RegExrNFA
from src.dfa_utils import nfa_to_dfa, simplify_dfa
from src.nfa import NFA

parser = lark.Lark(
    grammar='''
// the priority here is closure > concat > union
?regs : union
?union : concat "|" union | concat
?concat : closure concat | closure
?closure : elem "*" | elem
?elem : alph | "(" regs ")"
alph : "0" | "1" | "ε"
''',
    start='regs',
    keep_all_tokens=True,
)


def convert_tree(node: lark.ParseTree) -> ParseTree:
    def recursive(node: lark.ParseTree | lark.Token) -> ParseTree.Node:
        if isinstance(node, lark.Token):
            return ParseTree.Node(node)
        p_node = ParseTree.Node(node.data)
        for ch in node.children:
            # where is the god damn terminals ?
            p_node.children.append(recursive(ch))
        return p_node

    root_node = recursive(node)
    return ParseTree(root_node)


def parse_tree_to_nfa(tree: ParseTree) -> RegExrNFA:
    def recursive(node: ParseTree.Node) -> RegExrNFA:
        c = node.children
        r: RegExrNFA | None = None
        if node.label == 'alph':
            token = c[0].label
            if token == Epsilon:
                r = epsilon_exr_nfa()
            else:
                r = alphabet_exr_nfa(token)
        elif node.label == 'elem':
            assert len(c) == 3 and c[0].label == '(' and c[2].label == ')'
            r = recursive(c[1])
        elif node.label == 'concat':
            assert len(c) == 2
            r = concat(recursive(c[0]), recursive(c[1]))
        elif node.label == 'closure':
            assert len(c) == 2 and c[1].label == '*'
            r = closure(recursive(c[0]))
        elif node.label == 'union':
            assert len(c) == 3 and c[1].label == '|'
            r = union(recursive(c[0]), recursive(c[2]))
        else:
            raise RuntimeError(f'unsupported node: {node}')
        if r is None:
            raise RuntimeError(f'parse error!')
        return r

    return recursive(tree.root)


@dataclass
class Testcase:
    name: str
    value: str = ''
    load_nfa: bool = False

    def run(self):
        print(f'running {self.name}')
        if self.load_nfa:
            with open(f'{self.name}.json','r') as f:
                nfa = json5.load(f,object_hook=NFA.from_dict)
        else:
            assert self.value != ''
            parse_tree = convert_tree(parser.parse(self.value))
            # visualize parse tree
            graph = parse_tree.visualize()
            graph.format = 'svg'
            parse_tree_filename = f'{self.name}.parse_tree'
            graph.render(cleanup=True, filename=parse_tree_filename)
            print(f'svg file saved to {parse_tree_filename}.svg')
            nfa = parse_tree_to_nfa(parse_tree)
            
        # visualize nfa
        graph = nfa.visualize()
        graph.format = 'svg'
        nfa_filename = f'{self.name}.nfa'
        graph.render(cleanup=True, filename=nfa_filename)
        print(f'svg file saved to {nfa_filename}.svg')
        # visualize dfa
        dfa = nfa_to_dfa(nfa)
        graph = dfa.visualize()
        graph.format = 'svg'
        dfa_filename = f'{self.name}.dfa'
        graph.render(cleanup=True, filename=dfa_filename)
        print(f'svg file saved to {dfa_filename}.svg')
        # visualize simplified dfa
        sim_dfa = simplify_dfa(dfa)
        graph = sim_dfa.visualize()
        graph.format = 'svg'
        sim_dfa_filename = f'{self.name}.sim_dfa'
        graph.render(cleanup=True, filename=sim_dfa_filename)
        print(f'svg file saved to {sim_dfa_filename}.svg')


def main():
    """
    '1(0|1)*101',
    '1(1010*|1(010)*1)*0',
    '0*10*10*10*',
    '(00|11)*((01|10)(00|11)*(01|10)(00|11)*)*',
    # 包含 010 子串的语句
    '(0|1)*010(0|1)*',
    # 不包含 010 子串的语句
    '(1|00*11)*(ε|00*|00*1)',
    '1*(0|111*)*1*'
    """
    testcase = [
        Testcase('2-3.1', '1(0|1)*101'),
        Testcase('2-3.2', '1(1010*|1(010)*1)*0'),
        Testcase('2-3.3', '0*10*10*10*'),
        Testcase('2-3.4', '(00|11)*((01|10)(00|11)*(01|10)(00|11)*)*'),
        Testcase('2-4.1', '(0|1)*010(0|1)*'),
        Testcase('2-4.2', '1*(0|111*)*1*'),
        Testcase('2-5.a',load_nfa=True),
        Testcase('2-5.b',load_nfa=True)
    ]

    for t in testcase:
        print('='*50)
        t.run()
        print('='*50)
        print()


if __name__ == '__main__':
    main()