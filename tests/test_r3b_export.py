"""Figure-data extraction at selected replica counts and span dimensions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paper_experiments/extract_r3b_paper_data.py"
spec = importlib.util.spec_from_file_location("extract_r3b_paper_data", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.fixture
def summary():
    cells = {}
    for ji, J in enumerate((32, 32768)):
        for mi, m in enumerate((2, 8)):
            value = 10 * ji + mi
            cells[f"J{J}|m{m}|sampled"] = {
                "n": 4,
                "covered_all": 4,
                "any_violation": 0,
                "unresolved": 0,
                "ensemble_invariant": False,
                "B_median": [value, value + 0.5],
                "cost_last": {"modeled_seconds": value + 1.0},
            }
    mean = np.arange(8).reshape(2, 2, 2, 1).tolist()
    return {
        "setup": {
            "t_grid": [0, 1],
            "J_grid": [32, 32768],
            "m_grid": [2, 8],
            "N_grid": [16],
            "r_init": 0.8,
            "b_t": [0.1, 0.1],
            "b_pi": 0.1,
            "e_tN": [[0.2], [0.2]],
        },
        "n_ensembles": 4,
        "run_status": "COMPLETED",
        "truth": {"D": [0.3, 0.2], "L": [0.2, 0.1], "U": [0.4, 0.3], "b_ref_t": 0.1, "beta_ref": 0.01},
        "joint_guaranteed_sampled": {"covered_all": 4},
        "embedding_status": {"kind": "closed form"},
        "corrections": {"quad_margin_known": 0.0, "omega_max": 0.0, "inflation_max": 1.0, "methods": {}},
        "methods": {"B3_N16": cells},
        "labels": {"B3_N16": "guarantee"},
        "paired": {
            "B3-B4": {
                "mean": mean,
                "se": np.zeros((2, 2, 2, 1)).tolist(),
                "min": 0.0,
                "argmin": {},
                "frac_negative_cells": 0.0,
                "n_entries": 8,
                "n_negative": 0,
                "n_entries_t_pos": 4,
                "n_negative_t_pos": 0,
                "min_t_pos": 4.0,
                "n_negative_last_t": 0,
                "n_negative_t_pos_1e5": 0,
                "n_negative_last_t_1e5": 0,
                "n_negative_by_t_1e5": {},
            }
        },
        "transition_matched": [{"mean_B3_minus_B4": [[[1.0], [2.0]], [[3.0], [4.0]]], "se": []}],
        "cost": {},
    }


@pytest.mark.parametrize("J,m,ji,mi", [(32, 2, 0, 0), (32, 8, 0, 1), (32768, 2, 1, 0), (32768, 8, 1, 1)])
def test_compact_uses_requested_cell(summary, J, m, ji, mi):
    result = export.compact(summary, J, m)
    cell = summary["methods"]["B3_N16"][f"J{J}|m{m}|sampled"]
    selected = result["methods"]["B3_N16"]["primary"]
    assert selected["B_median"] == cell["B_median"]
    assert selected["cost_last"] == cell["cost_last"]
    assert result["paired"]["B3-B4"]["mean_primary"] == [2 * ji + mi, 4 + 2 * ji + mi]
    assert result["transition_matched"][0]["mean_B3_minus_B4_last_primary"] == 3.0 + mi
