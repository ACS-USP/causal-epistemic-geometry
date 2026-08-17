from epistemic_geometry.benchmarks.e3.splits import FAMILY_CELLS
from epistemic_geometry.benchmarks.e3.validation import validate_family


def test_cpu_generator_validation_covers_all_families() -> None:
    for family, cells in FAMILY_CELLS.items():
        report = validate_family(family, cells, n_per_cell=8)
        assert set(report) == set(cells)
        assert all(cell_report["unique_latents"] == 8 for cell_report in report.values())
