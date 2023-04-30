from __future__ import annotations
import lark
from src.parse_tree import ParseTree
from dataclasses import dataclass
from prettytable import PrettyTable


@dataclass
class BNF:
    definition: str
    start: str

    def get_parser(self) -> lark.Lark:
        return lark.Lark(grammar=self.definition,
                         start=self.start,
                         keep_all_tokens=True)


parser0 = BNF(
    definition='''
n : d | n d
d : "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
''',
    start='n',
).get_parser()

parser1 = BNF(
    definition='''
e : t | e "+" t | e "-" t 
t : f | t "*" f | t "/" f
f : "(" e ")" | "i"
''',
    start='e',
).get_parser()

def convert_tree(node: lark.ParseTree) -> ParseTree:
    def recursive(node: lark.ParseTree | lark.Token) -> ParseTree.Node:
        if isinstance(node, lark.Token):
            return ParseTree.Node(node.value)
        p_node = ParseTree.Node(node.data.capitalize())
        for ch in node.children:
            p_node.children.append(recursive(ch))
        return p_node

    root_node = recursive(node)
    return ParseTree(root_node)


def get_leftmost_derivation(tree: ParseTree) -> list[list[str]]:
    res: list[list[str]] = [[tree.root.label]]

    def recursive(node: ParseTree.Node, pos: int = 0) -> int:
        if len(node.children) == 0:
            return 0

        nonlocal res
        derivation = [child.label for child in node.children]
        res.append(res[-1][:pos] + derivation + res[-1][pos + 1:])
        n_children = len(node.children)
        right_offset = 0
        for i, child in enumerate(node.children):
            right_offset += recursive(child, right_offset + pos + i)
        right_offset += n_children - 1
        return right_offset

    recursive(tree.root)
    return res


def get_rightmost_derivation(tree: ParseTree) -> list[list[str]]:
    res: list[list[str]] = [[tree.root.label]]

    def recursive(node: ParseTree.Node, pos: int = 0):
        if len(node.children) == 0:
            return

        nonlocal res
        derivation = [child.label for child in node.children]
        res.append(res[-1][:pos] + derivation + res[-1][pos + 1:])
        n_children = len(node.children)
        for i, child in list(zip(range(n_children), node.children))[::-1]:
            recursive(child, pos + i)

    recursive(tree.root)
    return res


def format_derivation(derivations: list[list[str]]) -> str:
    assert len(derivations[0]) == 1
    res = ''
    res += derivations[0][0]
    for d in derivations[1:]:
        res += f'\n-> {"".join(d)}'
    return res


@dataclass
class Testcase:
    name: str
    parser: lark.Lark
    value: str
    gen_derivation: bool
    vis_parse_tree: bool

    def run(self):
        print(f'running {self.name}')
        parse_tree = convert_tree(self.parser.parse(self.value))
        if self.gen_derivation:
            ld = get_leftmost_derivation(parse_tree)
            rd = get_rightmost_derivation(parse_tree)
            left = format_derivation(ld)
            right = format_derivation(rd)
            table = PrettyTable(['Left derivation','Right derivation'])
            table.align = 'l'
            table.add_row([left,right])
            print(table)
        if self.vis_parse_tree:
            graph = parse_tree.visualize()
            graph.name = self.name
            graph.format = 'svg'
            graph.render(cleanup=True,filename=self.name)
            print(f'svg file saved to {self.name}.svg')


def main():
    testcase = [
        Testcase(
            name='1-2.2.1',
            parser=parser0,
            value='0127',
            gen_derivation=True,
            vis_parse_tree=False,
        ),Testcase(
            name='1-2.2.2',
            parser=parser0,
            value='34',
            gen_derivation=True,
            vis_parse_tree=False,
        ),Testcase(
            name='1-2.2.3',
            parser=parser0,
            value='568',
            gen_derivation=True,
            vis_parse_tree=False,
        ),Testcase(
            name='1-4.1.1',
            parser=parser1,
            value='i+i*i',
            gen_derivation=True,
            vis_parse_tree=False,
        ),Testcase(
            name='1-4.1.2',
            parser=parser1,
            value='i*(i+i)',
            gen_derivation=True,
            vis_parse_tree=False,
        ),Testcase(
            name='1-4.2.1',
            parser=parser1,
            value='i+i+i',
            gen_derivation=False,
            vis_parse_tree=True,
        ),Testcase(
            name='1-4.2.2',
            parser=parser1,
            value='i+i*i',
            gen_derivation=False,
            vis_parse_tree=True,
        ),Testcase(
            name='1-4.2.3',
            parser=parser1,
            value='i-i-i',
            gen_derivation=False,
            vis_parse_tree=True,
        ),
    ]

    for t in testcase:
        print('='*50)
        t.run()
        print('='*50)
        print()


if __name__ == '__main__':
    main()
