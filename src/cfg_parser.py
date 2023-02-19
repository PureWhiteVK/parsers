from ply.lex import Lexer, lex
from ply.yacc import LRParser, yacc
from typing import List, Optional, Set, cast
from src.cfg import CFG, Symbol, Terminal, Nonterminal, Production
from src.utils import Epsilon, check_type, Token, Rule

# multiline comment: r'[/][*][^*]*[*]+(?:[^/*][^*]*[*]+)*[/]' with re.MULTILINE flags on
# oneline comment: [/][/].* or [#].*


# data carried when parsing
class Payload:

    def __init__(self,
                 symbol: Symbol = '',
                 symbol_lists: Optional[List[Symbol]] = None,
                 body: Optional[List[List[Symbol]]] = None) -> None:
        self.symbol = symbol
        self.symbol_lists = [] if symbol_lists is None else symbol_lists
        self.body = [] if body is None else body

    def __repr__(self) -> str:
        return f'<symbol={self.symbol},symbol_lists={self.symbol_lists},body={self.body}>'


class CfgParser:

    def __init__(self) -> None:
        # print(dir(self))
        self.terminals: Set[Terminal] = set()
        self.nonterminals: Set[Nonterminal] = set()
        self.productions: Set[Production] = set()
        self.start_symbol: Nonterminal = ''

        self.lexer: Lexer = lex(object=self)
        self.parser: LRParser = yacc(module=self, start='production_list')

    tokens = ('NONTERMINAL', 'TERMINAL', 'VERTLINE', 'COLON', 'EPSILON',
              'SEMICOLON')

    # Tokens
    t_SEMICOLON = r';'
    t_VERTLINE = r'\|'
    t_COLON = r':'
    # using . to represent ε (for ascii compability)
    t_EPSILON = r'ε|\.'

    # ignore whitespace and horizontal tabulate character
    t_ignore = " \t"

    # 调用 t_newlinw 等价于 TOKEN(regex)(t_newline)()
    @Token(r'(?:(?:\n\r)|(?:\n))+')
    def t_newline(self, t):
        # \n\r 必须在最前面，否则会优先匹配 \n 导致后面的 \r 变成错误
        t.lineno += t.value.count("\n")

    @Token(r'[a-zA-Z_][_a-zA-Z0-9\']*')
    def t_NONTERMINAL(self, t):
        self.nonterminals.add(t.value)
        return t

    @Token(r'[\'\"][^\"\'\s]+[\'\"]')
    def t_TERMINAL(self, t):
        # 去除两端的引号
        t.value = t.value[1:-1]
        self.terminals.add(t.value)
        return t

    def t_error(self, t):
        # print('error??')
        # print(type(t))
        print(
            f'Illegal string {t.value[:10] + "..."!r} at line {t.lineno}, col {t.lexpos},\nskip 1 character to continue.'
        )
        t.lexer.skip(1)

    def p_error(self, p):
        raise SyntaxError(f'Syntax error at {p.value!r} at line {p.lineno}, col {p.lexpos}.')

    @Rule('''
    production_list : production 
                    | production production_list
    ''')
    def p_production_list(self, p):
        # parse finished!
        # return CFG
        p[0] = CFG(self.terminals, self.nonterminals, self.start_symbol,
                   self.productions)

    @Rule('''
    production : NONTERMINAL COLON body SEMICOLON
    ''')
    def p_production(self, p):
        if self.start_symbol == '':
            self.start_symbol = p[1]
        p3 = cast(Payload, p[3])
        for b in p3.body:
            self.productions.add(Production(p[1], b))

    @Rule('''
    body : EPSILON 
         | symbol_list 
         | symbol_list VERTLINE body
    ''')
    def p_body(self, p):
        if len(p) == 2:
            if p.slice[1].type == 'EPSILON':
                self.terminals.add(Epsilon)
                p[0] = Payload(body=[[]])
            else:
                p1 = cast(Payload, p[1])
                p[0] = Payload(body=[p1.symbol_lists])
        elif len(p) == 4:
            p1 = cast(Payload, p[1])
            p3 = cast(Payload, p[3])
            p3.body.insert(0, p1.symbol_lists)
            p[0] = p3

    @Rule('''
    symbol_list : symbol 
                | symbol symbol_list
    ''')
    def p_symbol_list(self, p):
        # node = ParseTree.Node('symbol_list')
        # if len(p) == 2:
        #     node.children = [p[1]]
        # elif len(p) == 3:
        #     node.children = [p[1],*p[2].children]
        # p[0] = node
        if len(p) == 2:
            check_type(p[1], Payload, 'p[1]')
            p1 = cast(Payload, p[1])
            p[0] = Payload(symbol_lists=[p1.symbol])
        elif len(p) == 3:
            check_type(p[1], Payload, 'p[1]')
            check_type(p[2], Payload, 'p[2]')
            p1 = cast(Payload, p[1])
            p2 = cast(Payload, p[2])
            p2.symbol_lists.insert(0, p1.symbol)
            p[0] = p2

    @Rule('''
    symbol : NONTERMINAL 
           | TERMINAL
    ''')
    def p_symbol(self, p):
        # node = ParseTree.Node('symbol')
        # node.children = [ParseTree.Node(f'{p.slice[1].type}: {p[1]}')]
        # p[0] = node
        check_type(p[1], Symbol, 'p[1]')
        p[0] = Payload(symbol=p[1])

    def parse(self, input: str) -> Optional[CFG]:
        return self.parser.parse(input, self.lexer)

    
def main():
    test_cfg = \
    """
    production_list : production 
                    | production production_list
                    ;
    production : NONTERMINAL ',' body ';' ;
    body : 'regex-for-epsilon' 
        | symbol_list 
        | symbol_list '|' body
        ;
    symbol_list : symbol 
                | symbol symbol_list
                ;
    symbol : NONTERMINAL 
        | TERMINAL
        ;
    NONTERMINAL : 'regex-for-nonterminal' ;
    TERMINAL : 'regex-for-terminal' ;
    """

    cfg_parser = CfgParser()

    # Build the lexer


    test_str = \
    """
    E : E '+' T 
    | E '-' T 
    | T 
    ;
    T : T '*' F 
    | T '/' F 
    | F 
    ;
    F : '(' E ')' 
    | 'id' 
    ;
    """

    cfg: Optional[CFG] = cfg_parser.parse(test_cfg)

    if cfg is None:
        exit(-1)

    print('parse finished')
    cfg.to_file('output_cfg.txt')

    with open('output_cfg.txt','r',encoding='utf-8') as f:
        cfg: Optional[CFG] = cfg_parser.parse(f.read())
        if cfg is None:
            exit(-1)

        print('parse finished')
        print(cfg)

if __name__ == '__main__':
    main()
