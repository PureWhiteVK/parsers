from os import path

ROOT_DIR = path.dirname(path.dirname(__file__))

OUTPUT_DIR = path.join(ROOT_DIR, 'output')

Epsilon: str = 'ε'

# empty set
Oslash: str = 'Ø'

# finish symbol
Dollar: str = '$'

# Invalid Symbol (just for safe test)
Invalid: str = ''

from typing import Callable, Collection, List, NamedTuple
from typing_extensions import TypeAlias

State: TypeAlias = str


def state_name(state: State):
    return f'{state}'


Input: TypeAlias = str


class TransPair(NamedTuple):
    current: State
    input: Input


class StatePair(NamedTuple):
    current: State
    target: State


class NameAllocator:

    def __init__(self, prefix='') -> None:
        self._id = 0
        self._prefix = prefix
        self._names: List[str] = []

    @property
    def next(self) -> str:
        _id = f'{self._prefix}{self._id}'
        self._names.append(_id)
        self._id += 1
        return _id

    @property
    def names(self) -> List[str]:
        return self._names


class IntAllocator:

    def __init__(self) -> None:
        self._id = 0
        self._ids: List[int] = []

    @property
    def next(self) -> int:
        _id = self._id
        self._ids.append(_id)
        self._id += 1
        return _id

    @property
    def ids(self) -> List[int]:
        return self._ids


def int_to_alphabet(v: int) -> str:
    if not hasattr(int_to_alphabet, 'number_map'):
        number_map: List[str] = [chr(i + ord('A')) for i in range(26)]
        number_map.insert(0, 'None')
        setattr(int_to_alphabet, 'number_map', number_map)
    number_map = getattr(int_to_alphabet, 'number_map')
    # state ranges from 0, and first we have to convert to 1 based
    #
    # convert state to Base 26 system
    s = v + 1
    # convert s to Base 26 system (but with no zero?)
    # A = 1, B = 2, ...
    res: List[str] = []
    while s > 0:
        v = int(s % 26)
        s = int(s / 26)
        if v == 0:
            v = 26
            s -= 1
        res.insert(0, number_map[v])
    return ''.join(res)


def check_type(_obj: object, _type, field_name: str):
    assert isinstance(
        _obj, _type
    ), f'Field {field_name} must be type {_type}, requested {type(_obj)}.'


def check_array_type(_list: Collection,
                     element_type,
                     list_type,
                     field_name: str,
                     allow_empty=False):
    check_type(_list, list_type, field_name)
    if not allow_empty:
        assert len(_list) != 0, f'Field {field_name} must not be empty.'
    assert all(
        isinstance(element, element_type) for element in _list
    ), f'Field {field_name} must be type List[{element_type}], requested List[{[type(element) for element in _list]}].'


def is_array_type(_list: Collection, element_type, list_type):
    return isinstance(_list, list_type) and all(
        isinstance(element, element_type) for element in _list)


def Token(regex: str) -> Callable:
    '''
    add regex: str to function's regex attribute
    '''

    def add_regex(f: Callable) -> Callable:
        assert hasattr(
            f, '__call__'), f'@Token must be applied to Callable object'
        setattr(f, 'regex', regex)
        return f

    return add_regex


def Rule(rule: str) -> Callable:
    '''
    add rule: str to function's __doc__ attribute
    '''

    def add_rule(f: Callable) -> Callable:
        assert hasattr(f,
                       '__call__'), f'@Rule must be applied to Callable object'
        setattr(f, '__doc__', rule)
        return f

    return add_rule
