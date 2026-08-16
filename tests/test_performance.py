from epistemic_geometry.inference.performance import ModeMeasurement, compare_modes, cost_summary


def test_cost_model_reports_throughput_and_speedup() -> None:
    slow = ModeMeasurement("serial_reference", 10.0, 100)
    fast = ModeMeasurement("cached_decode", 2.0, 100)
    assert cost_summary(slow)["seconds_per_1k_item_condition"] == 100.0
    rows = compare_modes([slow, fast])
    assert rows[0]["speedup_vs_reference"] == 1.0
    assert rows[1]["speedup_vs_reference"] == 5.0
