"""Read-only, hash-validating loaders for the frozen Q2 V4.1 publication sources."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
FIGURE_SPEC_PATH = ROOT / "manuscript/figures/paper1_q2/FIGURE_SPEC.json"
FORBIDDEN_PARTS = {
    "livecodebench",
    "q1_second_task",
    "q1-second-task",
    "q3",
    "journal.jsonl",
    "semantic_scores.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def figure_spec() -> dict[str, Any]:
    return json.loads(FIGURE_SPEC_PATH.read_text(encoding="utf-8"))


def expected_source_hashes() -> dict[str, str]:
    return dict(figure_spec()["expected_source_sha256"])


def _safe_path(relative_path: str) -> Path:
    relative = Path(relative_path)
    lowered = {part.lower() for part in relative.parts}
    if lowered & FORBIDDEN_PARTS or any(
        token in relative_path.lower()
        for token in ("livecodebench", "q1_second_task", "q1-second-task", "/q3")
    ):
        raise RuntimeError("Q2 publication firewall rejected an out-of-scope source")
    path = (ROOT / relative).resolve()
    if ROOT not in path.parents:
        raise RuntimeError("publication source escaped repository root")
    return path


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads(_safe_path(relative_path).read_text(encoding="utf-8"))


def load_numpy(relative_path: str) -> np.ndarray | dict[str, np.ndarray]:
    loaded = np.load(_safe_path(relative_path), allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            return {key: np.asarray(loaded[key]) for key in loaded.files}
        finally:
            loaded.close()
    return np.asarray(loaded)


def validate_frozen_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path, expected in expected_source_hashes().items():
        path = _safe_path(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen Q2 source: {relative_path}")
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"frozen Q2 source hash mismatch for {relative_path}: {actual} != {expected}"
            )
        observed[relative_path] = actual
    return observed


def _validate_scientific_state(data: dict[str, Any]) -> None:
    estimands = data["estimands"]
    radial = data["radial"]
    closeout = data["closeout"]
    forensic = data["forensic"]
    completeness = data["completeness"]
    spec = data["spec"]

    if estimands.get("classification") != "Q2_V4_1_G2":
        raise RuntimeError("Q2 terminal classification changed")
    if closeout.get("classification") != "Q2_V4_1_G2":
        raise RuntimeError("Q2 closeout classification changed")
    if radial["R_shape"].get("classification") != "RS+":
        raise RuntimeError("Q2 radial shape classification changed")
    if radial["R_total"].get("classification") != "RT+":
        raise RuntimeError("Q2 radial total classification changed")
    if forensic.get("classification") != "Q2_V4_1_SEMANTIC_FORENSIC_CLEAN":
        raise RuntimeError("Q2 forensic classification changed")
    if forensic.get("maximum_difference") != 0.0:
        raise RuntimeError("Q2 forensic audit no longer reconciles exactly")
    if estimands.get("q3") != "NOT_RUN" or closeout.get("resources", {}).get("q3") != "NOT_RUN":
        raise RuntimeError("Q2 package must not contain a Q3 result")

    expected_rows = 37_800
    if completeness.get("expected_logical_rows") != expected_rows:
        raise RuntimeError("Q2 expected trajectory count changed")
    if completeness.get("observed_logical_rows") != expected_rows:
        raise RuntimeError("Q2 observed trajectory count changed")
    for field in ("missing", "unexpected", "duplicates", "replacements"):
        if completeness.get(field) not in (0, []):
            raise RuntimeError(f"Q2 campaign integrity field {field} is nonzero")

    controllers = list(estimands["controller_order"])
    frozen_order = list(data["safe_bank"]["candidate_order"])
    expected_count = int(spec["global_rules"]["controller_count"])
    if controllers != frozen_order or len(controllers) != expected_count:
        raise RuntimeError("Q2 controller identity/order mismatch")
    if len(set(controllers)) != expected_count:
        raise RuntimeError("Q2 controller IDs are not unique")

    matrices = data["matrices"]
    expected_keys = {
        f"{metric}_{shell}" for metric in ("A0", "A1", "A2") for shell in ("MEDIUM", "STRONG")
    }
    if not expected_keys <= matrices.keys():
        raise RuntimeError("Q2 prediction matrix archive is incomplete")
    for key in expected_keys:
        matrix = np.asarray(matrices[key], dtype=np.float64)
        if matrix.shape != (expected_count, expected_count):
            raise RuntimeError(f"Q2 matrix {key} has wrong shape")
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
            raise RuntimeError(f"Q2 matrix {key} is not finite symmetric")

    dshape = estimands["semantic_distance"]["D_shape_superpopulation"]
    for shell in ("MEDIUM", "STRONG"):
        matrix = np.asarray(dshape[shell], dtype=np.float64)
        if matrix.shape != (expected_count, expected_count):
            raise RuntimeError(f"Q2 D_shape {shell} has wrong shape")
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T):
            raise RuntimeError(f"Q2 D_shape {shell} is not finite symmetric")

    permutations = np.asarray(data["qap_permutations"])
    if permutations.shape != (50_000, expected_count):
        raise RuntimeError("frozen QAP schedule shape changed")
    if not np.array_equal(permutations[0], np.arange(expected_count)):
        raise RuntimeError("frozen QAP schedule no longer starts with identity")


def load_sources() -> dict[str, Any]:
    source_hashes = validate_frozen_sources()
    data: dict[str, Any] = {
        "source_hashes": source_hashes,
        "spec": figure_spec(),
        "estimands": load_json("review/q2_v4_1_semantic_execution/ESTIMANDS.json"),
        "bootstrap": load_json("review/q2_v4_1_semantic_execution/BOOTSTRAP_INTERVALS.json"),
        "radial": load_json("review/q2_v4_1_semantic_execution/RADIAL_RESULTS.json"),
        "closeout": load_json("review/q2_v4_1_semantic_execution/Q2_V4_1_SEMANTIC_CLOSEOUT.json"),
        "forensic": load_json("review/q2_v4_1_semantic_execution/FORENSIC_AUDIT.json"),
        "completeness": load_json("review/q2_v4_1_semantic_execution/CAMPAIGN_COMPLETENESS.json"),
        "posthoc": load_json(
            "review/q2_v4_1_semantic_execution/POST_HOC_GENERATION_DIAGNOSTIC.json"
        ),
        "safe_bank": load_json(
            "review/q2_v4_1_31_safe_bank_review/SAFE_31_IMMUTABLE_MANIFEST.json"
        ),
        "candidate_bank": load_json("review/q2_v4_spark1_presemantic/CANDIDATE_BANK_MANIFEST.json"),
        "subspace": load_json("review/q2_v4_spark1_presemantic/SPARK1_SUBSPACE_QUALIFICATION.json"),
        "matrix_metadata": load_json(
            "review/q2_v4_1_prediction_lock/PREDICTION_MATRIX_METADATA.json"
        ),
        "matrices": load_numpy("review/q2_v4_1_prediction_lock/PREDICTION_MATRICES.npz"),
        "qap_permutations": load_numpy(
            "review/q2_v4_1_prediction_lock/QAP_CONTROLLER_PERMUTATIONS.npy"
        ),
        "qap_schedule": load_json("review/q2_v4_1_prediction_lock/QAP_SCHEDULE.json"),
    }
    _validate_scientific_state(data)
    return data


__all__ = [
    "FIGURE_SPEC_PATH",
    "ROOT",
    "expected_source_hashes",
    "figure_spec",
    "load_sources",
    "sha256",
    "validate_frozen_sources",
]
