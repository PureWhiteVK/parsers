# Automata and Languages

## Feature

**Data Structure** 

support graphviz visualization and JSON serialization

- [x] NFA、DFA
- [x] Context Free Grammar
- [x] Parse Tree、Trie

**Algorithms**

- [x] Automata operation

  - NFA to DFA （subset construction）

  - DFA Minimizing 
  - Regular Expression (parse tree) to NFA

- [x] CFG operation 

  - left factoring

  - eliminate left recursion

  - eliminate useless symbol

  - eliminate epsilon production

  - eliminate unit production

  - CFG to CNF

- [x] Syntax analysis

  - CYK

  - LL(1) 

  - Operator precedence

  - SLR

**Simple notation for BNF / CFG** 

with a tiny parser implemented with ply (python lex yacc)

```
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
```

example

***sample.txt***

```
E : E '+' T | E '-' T | T ;
T : T '*' F | T '/' F | F ;
F : '(' E ')' | 'id' ;
```



## Examples



## Reference

1. [COMS W3261 Computer Science Theory Sect 001 (columbia.edu)](http://www.cs.columbia.edu/~aho/cs3261/)
