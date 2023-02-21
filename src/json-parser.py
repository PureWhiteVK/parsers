from typing import List, Union
from lark import Lark, Transformer
from lark.lexer import Token
from pathlib import Path
from time import perf_counter_ns

# a json is just identical to Dict[str,Value]
Value = Union[str, float, bool, list, dict, None]


def recursive(m) -> str:
    if isinstance(m, list):
        return f'[ {",".join(recursive(i) for i in m)} ]'
    if isinstance(m, tuple):
        return f'( {",".join(recursive(i) for i in m)} )'
    return str(type(m))


class LarkJsonTransformer(Transformer):
    # we have to convert this to normal char
    def escaped_utf16(self, v: List[Token]) -> str:
        w0 = int(v[0], base=16)
        w1 = int(v[1], base=16)
        return bytes([w0 >> 8, w0 & 0xFF, w1 >> 8,
                      w1 & 0xFF]).decode('utf_16_be')

    def escaped_char(self, v: List[Token]) -> str:
        # print(f'escaped_char: {v}')
        return v[0].encode('ascii').decode('unicode_escape')

    def string(self, s: List[Token]) -> str:
        # print(f'string: {s}')
        # return ''.join(s[1:-1])
        return ''.join(s)

    def number(self, n: List[Token]) -> float:
        return float(n[0])

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False
    object = dict
    array = list
    member = tuple


root = Path(__file__).parent

example_path = root / "lark-json"

import json


class Timer:
    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        self.start = perf_counter_ns()

    def __exit__(self, type, value, trace):
        self.end = perf_counter_ns()
        print(f'[ {self.name} ] used: {(self.end-self.start) / 1e6 :.2f}ms')


# 7.82 MB (8,201,296 字节)
# 和json库相比至少是10倍的性能吧
def benchmark():
    benchmark_file = root / "generated.json"

    with open(benchmark_file, "r") as f:
        with Timer("json.load"):
            l = len(json.load(f))
        print(l)

        f.seek(0)
        with Timer("json.loads"):
            s = f.read()
            l = len(json.loads(s))
        print(l)

        transformer = LarkJsonTransformer(False)
        # test lark-json
        parser = Lark.open(example_path / "json.lark", rel_to=example_path)
        f.seek(0)
        with Timer("lark early"):
            s = f.read()
            l = len(transformer.transform(parser.parse(s)))
        print(l)

        f.seek(0)
        parser = Lark.open(example_path / "json.lark",
                           rel_to=example_path,
                           parser="lalr")
        with Timer("lark lalr(1)"):
            s = f.read()
            l = len(transformer.transform(parser.parse(s)))
        print(l)

        f.seek(0)
        parser = Lark.open(example_path / "json.lark",
                           rel_to=example_path,
                           parser="lalr",
                           transformer=LarkJsonTransformer(False))
        with Timer("lark lalr(1) treeless"):
            s = f.read()
            l = len(parser.parse(s))
        print(l)

    import subprocess
    import sys
    p = subprocess.Popen("python -VV", stdout=sys.stdout)
    exit(p.wait())


def main():
    transformer = LarkJsonTransformer()
    # test lark-json
    parser = Lark.open(example_path / "json.lark")
    with open(root / "a.json", "r", encoding="utf-8") as f:
        res = transformer.transform(parser.parse(f.read()))
    print(res)
    print(res['key'][5])
    # with open(root / "a.json","w",encoding="utf-8") as f:
    #     json.dump(res,f)
    # pass


if __name__ == "__main__":
    main()