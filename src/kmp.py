from typing import Mapping, NewType
from prettytable import PrettyTable
"""
compute failure function for keyword b1b2...bn

f and b are array start from 1, with length n

t = 0
f[1] = 0
// the s here could be interpreted as state in dfa
for (s = 1;s < n; s++) {
    while(t > 0 && b[s+1] != b[t+1]) {
        t = f[t]
    }
    if(b[s+1] == b[t+1]) {
        t = t + 1;
        f[s+1] = t;
    } else {
        f[s+1] = 0;
    }
}
"""


def main():

    def kmp_match(test: str, pattern: str) -> int:

        # make State different from int, for readability
        State = NewType('State', int)
        """
        return a int, indicate the index of start position
                      -1 means invalid
        """
        n = len(pattern)
        START_STATE = State(-1)
        ACCEPT_STATE = State(n - 1)

        # 此时我们的state是从-1到n-1，而不是从0到n

        def print_failure_fallback(f: Mapping[State, State]):
            # state range from -1 to n-1
            def state_name(s: State) -> str:
                return f'S{s+1}'

            table = PrettyTable(field_names=[
                's',
                state_name(START_STATE), *list(
                    state_name(s) for s in f.keys())
            ])
            table.add_row(
                ['f(s)', '\\', *list(state_name(s) for s in f.values())])
            print(table)

        def next_state(state: State) -> State:
            return State(state + 1)

        def get_input_symbol(state: State) -> str:
            return pattern[state + 1]

        # 这个代码写的实际上有点奇怪的，主要问题在于状态和索引之间没有很好的区分，导致代码看的不是特别懂
        def compute_failure_function() -> Mapping[State, State]:
            # temporary state
            t = START_STATE
            # failure_fallback[i] 表示状态 i-1 所指向的状态
            failure_fallback: Mapping[State, State] = dict([
                (next_state(START_STATE), START_STATE)
            ])
            # s is also a state,此处不需要考虑结束状态，因为其并不包含任何输入
            for s_id in range(next_state(START_STATE), ACCEPT_STATE):
                s = State(s_id)
                while t != START_STATE and get_input_symbol(
                        t) != get_input_symbol(s):
                    t = failure_fallback[t]
                if get_input_symbol(t) == get_input_symbol(s):
                    t = next_state(t)
                    failure_fallback[next_state(s)] = t
                else:
                    failure_fallback[next_state(s)] = START_STATE
            return failure_fallback

        # first compute failure function
        f = compute_failure_function()
        print(len(f))
        print_failure_fallback(f)

        # then perform matching operation
        current_state = START_STATE
        m = len(test)
        for i in range(m):
            while current_state != START_STATE and test[i] != get_input_symbol(
                    current_state):
                current_state = f[current_state]
            if test[i] == get_input_symbol(current_state):
                current_state = next_state(current_state)
            if current_state == ACCEPT_STATE:
                return i - n + 1
        return -1

    def brute_force_match(test: str, pattern: str) -> int:
        # 直接遍历所有长度为m的子串，逐一进行比较
        n = len(pattern)
        m = len(test)
        for start_pos in range(m - n + 1):
            match = True
            for i in range(n):
                if pattern[i] != test[start_pos + i]:
                    match = False
                    break
            if match:
                return start_pos
        return -1

    print(kmp_match("abababaab", "ababaa"))
    # print(brute_force_match("abababaab","ababaa"))
    # print(kmp_match("abababaa","ababaa"))
    # print(brute_force_match("abababaa","ababaa"))


if __name__ == "__main__":
    main()