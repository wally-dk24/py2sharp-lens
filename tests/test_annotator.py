"""One test per catalog construct: a small snippet must produce the right
key at the right line number.  Each case is individually identifiable by
its construct key (the parametrize id)."""

import pytest

from app.annotator import annotate

# (construct_key, snippet, expected_line)
CASES = [
    ("ListComp", "squares = [i*i for i in range(5)]", 1),
    ("DictComp", "d = {k: k*k for k in range(3)}", 1),
    ("SetComp", "s = {x for x in items}", 1),
    ("GenExp", "total = sum(x*x for x in items)", 1),
    ("Yield", "def gen():\n    yield 1", 2),
    ("With", 'with open("f") as fh:\n    data = fh.read()', 1),
    ("Decorator", "@functools.cache\ndef fib(n):\n    return n", 2),
    ("StarArgs", "def f(*args, **kwargs):\n    pass", 1),
    ("FString", 's = f"hello {name}"', 1),
    ("Slice", "x = items[1:3]", 1),
    ("NegIndex", "x = items[-1]", 1),
    ("TupleUnpack", "a, b = 1, 2", 1),
    ("FloorDiv", "x = -7 // 2", 1),
    ("Power", "y = x ** 2", 1),
    ("IsOp", "if x is None:\n    pass", 1),
    ("InOp", "if k in d:\n    pass", 1),
    ("Truthiness", "if items:\n    print(items)", 1),
    ("TryExcept", "try:\n    f()\nexcept ValueError:\n    pass", 1),
    ("Lambda", "f = lambda x: x * 2", 1),
    ("Dataclass", "@dataclass\nclass Point:\n    x: int\n    y: int", 2),
    ("Property", "class C:\n    @property\n    def name(self):\n        return self._n", 3),
    ("InitSelf", "class C:\n    def __init__(self):\n        pass", 2),
    ("StrRepr", "class C:\n    def __str__(self):\n        return 'c'", 2),
    ("Enumerate", "for i, v in enumerate(items):\n    print(i, v)", 1),
    ("Zip", "for a, b in zip(x, y):\n    print(a, b)", 1),
    ("Range", "for i in range(5):\n    print(i)", 1),
    ("AsyncDef", "async def fetch():\n    pass", 1),
    ("Await", "async def fetch():\n    return await get()", 2),
    ("Global", "def f():\n    global x\n    x = 1", 2),
    ("Nonlocal", "def outer():\n    x = 1\n    def inner():\n        nonlocal x", 4),
    ("NoneLit", "x = None", 1),
    ("Assert", "assert x > 0", 1),
    ("Del", "del items[0]", 1),
    ("Match", 'match cmd:\n    case "go":\n        run()', 1),
    ("TypeHints", "def add(a: int, b: int) -> int:\n    return a + b", 1),
    ("MainGuard", 'if __name__ == "__main__":\n    main()', 1),
    ("Walrus", "if (n := len(items)) > 0:\n    print(n)", 1),
    ("ChainedComp", "if 1 < x < 10:\n    print(x)", 1),
    ("ForElse", "for i in range(3):\n    pass\nelse:\n    print('done')", 1),
    ("DictGet", 'v = d.get("k", 0)', 1),
    ("StrJoin", 's = ", ".join(items)', 1),
    ("RaiseFrom", "try:\n    pass\nexcept Exception as e:\n    raise ValueError('x') from e", 4),
    ("MutableDefault", "def f(items=[]):\n    pass", 1),
    ("LateBinding", "funcs = []\nfor i in range(3):\n    funcs.append(lambda: i)", 3),
    ("Ternary", "v = x if c else y", 1),
    ("DictMerge", "d = {**a, **b}", 1),
    ("StarUnpack", "f(*args, **kwargs)", 1),
]


def _findings_for(code: str, key: str):
    return [f for f in annotate(code) if f["construct_key"] == key]


@pytest.mark.parametrize("key,code,line", CASES, ids=[c[0] for c in CASES])
def test_construct_detected(key: str, code: str, line: int):
    found = _findings_for(code, key)
    assert found, f"no finding for construct {key!r} in {code!r}"
    assert found[0]["line"] == line, (
        f"{key}: expected line {line}, got {found[0]['line']}"
    )


def test_end_line_defaults_to_line():
    found = _findings_for("x = -7 // 2", "FloorDiv")
    assert found[0]["end_line"] == 1


def test_findings_sorted_by_line():
    code = "squares = [i*i for i in range(5)]\nx = -7 // 2"
    lines = [f["line"] for f in annotate(code)]
    assert lines == sorted(lines)


def test_invalid_python_raises_syntax_error():
    with pytest.raises(SyntaxError):
        annotate("def broken(:")


def test_floor_div_and_list_comp_together():
    code = "x = -7 // 2\nsquares = [i*i for i in range(5)]"
    keys = {(f["construct_key"], f["line"]) for f in annotate(code)}
    assert ("FloorDiv", 1) in keys
    assert ("ListComp", 2) in keys
