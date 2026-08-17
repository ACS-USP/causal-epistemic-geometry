from epistemic_geometry.benchmarks.e3.fsm10 import oracle as fsm_oracle
from epistemic_geometry.benchmarks.e3.modreg10 import oracle as modreg_oracle
from epistemic_geometry.benchmarks.e3.reachcount10 import oracle as reach_oracle
from epistemic_geometry.benchmarks.e3.satcount10 import oracle as sat_oracle


def test_modreg_oracle_hand_case() -> None:
    spec = {
        "initial": [1, 2, 3, 4],
        "operations": [
            {"op": "ADD_CONST", "r": "R0", "c": 9},
            {"op": "MUL_UNIT", "r": "R0", "u": 3},
            {"op": "ADD_REG", "dst": "R0", "src": "R1"},
            {"op": "SWAP", "r1": "R0", "r2": "R3"},
        ],
        "query": "R3",
    }
    assert modreg_oracle(spec) == 2


def test_fsm_oracle_composes_permutations() -> None:
    spec = {
        "transitions": {
            "A": list(range(10)),
            "B": list(reversed(range(10))),
            "C": [(value + 1) % 10 for value in range(10)],
        },
        "start": 2,
        "sequence": ["C", "B", "C"],
    }
    assert fsm_oracle(spec) == 7


def test_reachcount_is_bounded_and_excludes_source() -> None:
    spec = {"edges": [[0, 1], [1, 2], [2, 3], [0, 4]], "source": 0, "max_hops": 2}
    assert reach_oracle(spec) == 3


def test_satcount_exhaustively_counts_then_reduces_modulo_ten() -> None:
    spec = {"n_variables": 2, "clauses": [[1], [-2]]}
    assert sat_oracle(spec) == 1
