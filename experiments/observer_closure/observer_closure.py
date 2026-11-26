"""Finite nonlinear observer pullback closure (outside 64 KiB core)."""
from dataclasses import dataclass
from vld import monoid_power, ExactFieldMatrix, myhill_nerode_quotient


def _pull(row, action):
    return sum(((row >> y) & 1) << x for x, y in enumerate(action))


def _compose(a, b):
    out = []
    for row in a:
        result = 0
        while row:
            j = row.bit_length() - 1
            row ^= 1 << j
            result ^= b[j]
        out.append(result)
    return tuple(out)


def apply(matrix, state):
    return sum(((row & state).bit_count() & 1) << i
               for i, row in enumerate(matrix))


@dataclass(frozen=True)
class Closure:
    size: int
    basis: tuple
    operators: tuple
    observer: int

    def encode(self, state):
        if type(state) is not int or not 0 <= state < self.size:
            raise ValueError("state outside domain")
        return sum(((row >> state) & 1) << i for i, row in enumerate(self.basis))

    def compile(self, word, repetitions):
        if type(repetitions) is not int or repetitions < 0:
            raise ValueError("invalid time")
        if any(type(a) is not int or not 0 <= a < len(self.operators) for a in word):
            raise ValueError("invalid action")
        identity = tuple(1 << i for i in range(len(self.basis)))
        block = identity
        for a in word:
            block = _compose(self.operators[a], block)
        powered = monoid_power(block, repetitions, _compose, identity)
        return powered

    def navigate(self, state, word, repetitions):
        return (self.observer & apply(self.compile(word,repetitions), self.encode(state))).bit_count() & 1


def form(actions, observer, rank_limit=256):
    if not actions or not 1 <= len(actions[0]) <= 256:
        raise ValueError("source size outside experiment scope")
    size = len(actions[0])
    if (any(len(a) != size or any(type(y) is not int or not 0 <= y < size for y in a)
            for a in actions) or type(observer) is not int or not 0 < observer < 1 << size
            or type(rank_limit) is not int or not 1 <= rank_limit <= 256):
        raise ValueError("invalid finite source")
    rows, pivots = [], {}

    def reduce(row):
        code = 0
        while row:
            p = row.bit_length() - 1
            if p not in pivots:
                break
            i = pivots[p]
            row ^= rows[i]
            code ^= 1 << i
        return row, code

    def add(row):
        row, _ = reduce(row)
        if row:
            if len(rows) >= rank_limit:
                raise OverflowError("UNRESOLVED: observer closure exceeds rank budget")
            pivots[row.bit_length()-1] = len(rows)
            rows.append(row)

    add(observer)
    i = 0
    while i < len(rows):
        for action in actions:
            add(_pull(rows[i], action))
        i += 1

    def express(row):
        rest, code = reduce(row)
        if rest:
            raise AssertionError("unclosed operator")
        return code

    return Closure(size, tuple(rows),
                   tuple(tuple(express(_pull(row, a)) for row in rows) for a in actions),
                   express(observer))


def verify(ground, actions, observer):
    """Independent closure/minimal-rank audit via native exact field RREF."""
    n = ground.size
    if not actions or any(len(a) != n for a in actions):
        return False
    def bits(row):
        return [(row >> i) & 1 for i in range(n)]
    def canonical(rows):
        mat, piv = ExactFieldMatrix(len(rows), n, rows, p=2).rref()
        return mat.data[:len(piv)]
    rows = canonical([bits(observer)])
    while True:
        more = canonical(rows + [[r[y] for y in a] for r in rows for a in actions])
        if len(more) == len(rows):
            break
        rows = more
    rank = len(rows)
    if (rank != len(ground.basis) or len(canonical([bits(r) for r in ground.basis])) != rank
            or len(ground.operators) != len(actions)
            or ground.observer >= 1 << rank or ground.observer < 0
            or any(len(m) != rank or any(not 0 <= r < 1 << rank for r in m)
                   for m in ground.operators)):
        return False
    for x in range(n):
        z = ground.encode(x)
        if ((ground.observer & z).bit_count() & 1) != ((observer >> x) & 1):
            return False
        if any(apply(m,z) != ground.encode(a[x]) for m,a in zip(ground.operators,actions)):
            return False
    # The generated span contains the observer and is closed; equality of
    # independently derived minimal rank implies equality of spans.
    return True


def fixtures():
    def nuisance(x):
        return (x & 15) | ((((x >> 4) + 1 + int((x & 3) == 3)) & 15) << 4)
    def rot(x):
        return (x & 240) | (((x & 15) << 1) & 15) | ((x >> 3) & 1)
    maps = tuple(tuple(fn(x) for x in range(256)) for fn in (
        lambda x: nuisance(rot(x)),
        lambda x: nuisance(x ^ (((x >> 1) & (x >> 2)) & 1)),
        lambda x: nuisance(x ^ (((x >> 4) & (x >> 2)) & 1)),
    ))
    observer = sum((x & 1) << x for x in range(256))
    return maps, observer


def quotient_count(actions, observer):
    q,_ = myhill_nerode_quotient(tuple(range(len(actions[0]))),
                                tuple(range(len(actions))),
                                lambda x,a:actions[a][x],
                                lambda x:(observer >> x)&1)
    return len(set(q.values()))
