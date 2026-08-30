from scripts.posthoc_diagnose_q2_v4_1_generation import is_extreme_mechanical_repetition


def test_short_sequences_are_not_flagged() -> None:
    assert not is_extreme_mechanical_repetition([1, 2, 3] * 80)


def test_periodic_tail_is_flagged() -> None:
    assert is_extreme_mechanical_repetition([1, 2, 3, 4] * 100)


def test_dominant_token_tail_is_flagged() -> None:
    assert is_extreme_mechanical_repetition(([1] * 200) + list(range(100)))


def test_diverse_tail_is_not_flagged() -> None:
    assert not is_extreme_mechanical_repetition(list(range(512)))
