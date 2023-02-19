from typing import List, Dict, Optional, Any, Union, cast
import json
from graphviz import Digraph

if __name__ == '__main__':
    # always shit here to make it available in both case
    import os
    import sys
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.utils import check_array_type

# python object for parse tree


class ParseTree:

    class Node:

        def __init__(
                self,
                label: str = '',
                children: Optional[List['ParseTree.Node']] = None) -> None:
            self.label = label
            # make sure self.children is not None
            self._children = [] if children is None else children
            self._count = len(self._children)

        def __repr__(self) -> str:
            return f'<label={self.label},children_count={self._count}>'

        @property
        def count(self):
            return self._count

        @count.setter
        def count(self, value):
            self._count = value

        @property
        def children(self):
            return self._children

        @children.setter
        def children(self, value: List['ParseTree.Node']):
            self._children = value
            self._count = len(self._children)

        @staticmethod
        def from_dict(data: Dict) -> 'ParseTree.Node':
            label = data.get('label')
            count = data.get('count')
            if not isinstance(label, str):
                raise TypeError(f'data.label must be str')
            if not isinstance(count, int):
                raise TypeError(f'data.children must be int')
            node = ParseTree.Node(label)
            node.count = count
            return node

        def to_dict(self) -> Dict[str, Any]:
            return {'type': 'Node', 'label': self.label, 'count': self.count}

    def __init__(self, root: 'ParseTree.Node') -> None:
        self._root = root

    @property
    def root(self):
        return self._root

    def visualize(self) -> Digraph:
        g = Digraph('parse_tree')
        node_id = 0

        def recursive(node: 'ParseTree.Node', parent_id: int = -1) -> None:
            nonlocal node_id
            current_id = node_id
            g.node(f'n{current_id}', label=node.label, shape='none')
            node_id += 1
            if parent_id != -1:
                g.edge(f'n{parent_id}', f'n{current_id}', dir='none')
            for child in node._children:
                recursive(child, current_id)

        recursive(self.root)
        return g

    def to_dict(self) -> Dict[str, Any]:
        nodes: List[Dict] = []
        queue: List['ParseTree.Node'] = [self.root]
        while len(queue) != 0:
            curr_node = queue.pop(0)
            queue.extend(curr_node.children)
            nodes.append(curr_node.to_dict())
        return {'type': 'ParseTree', 'nodes': nodes}

    @staticmethod
    def from_dict(
            data: Dict[str, Any]) -> Union['ParseTree', 'ParseTree.Node']:
        queue: List['ParseTree.Node'] = []
        type_name = cast(str, data.get('type'))
        if type_name == 'Node':
            return ParseTree.Node.from_dict(data)
        elif type_name != 'ParseTree':
            raise ValueError(
                f'type field of json object must be `Node` or `ParseTree`, requested: {type_name}'
            )
        nodes = cast(List['ParseTree.Node'], data.get('nodes'))
        check_array_type(nodes, ParseTree.Node, list, 'nodes')
        root: 'ParseTree.Node' = nodes[0]
        queue.append(root)
        pos = 1
        while len(queue) != 0:
            curr_node = queue.pop(0)
            curr_node.children = [
                nodes[pos + i] for i in range(curr_node.count)
            ]
            queue.extend(curr_node.children)
            pos += curr_node.count
        return ParseTree(root)


def main():

    def l(ch: str) -> ParseTree.Node:
        return ParseTree.Node(ch)

    def i(children: List[ParseTree.Node]) -> ParseTree.Node:
        return ParseTree.Node('$r', children)

    l0 = l('(')
    l1 = l('a')
    l2 = l('|')
    l3 = l('b')
    l4 = l(')')
    l5 = l('*')
    n0 = i([l1])
    n1 = i([l3])
    n2 = i([n0, l2, n1])
    n3 = i([l0, n2, l4])
    n4 = i([n3, l5])

    tree = ParseTree(n4)

    with open('test.json', 'w') as f:
        json.dump(tree.to_dict(), f)

    with open('test.json', 'r') as f:
        data = cast(ParseTree, json.load(f, object_hook=ParseTree.from_dict))
        # serialize
        s = json.dumps(data.to_dict())
        # deserialize
        d: ParseTree = cast(ParseTree,
                            json.loads(s, object_hook=ParseTree.from_dict))
        g = d.visualize()
        g.view(cleanup=True)


if __name__ == "__main__":
    main()
