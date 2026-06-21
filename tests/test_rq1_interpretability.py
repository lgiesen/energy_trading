from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.build_rq1_interpretability import (
    WarningRow,
    _collect_tft,
    _summary_rows,
    _write_latex_table,
    deterministic_top_n,
    feature_group,
)


def test_feature_group_mapper_representative_names() -> None:
    assert feature_group("da_price_lag_24h") == "price history"
    assert feature_group("load_forecast_da_entsoe_h2") == "load"
    assert feature_group("solar_forecast_da_entsoe_h2") == "renewable generation"
    assert feature_group("residual_load_forecast_da_h2") == "residual load"
    assert feature_group("afrr_capacity_price_neg_lag_1h") == "capacity market history"
    assert feature_group("afrr_activation_rate_pos_lag_1h") == "activation history"
    assert feature_group("unknown_feature") == "other"


def test_top_n_extraction_is_deterministic() -> None:
    df = pd.DataFrame(
        [
            {"target": "A", "model": "XGB", "importance_type": "shap", "feature": "b", "importance_value": 1.0},
            {"target": "A", "model": "XGB", "importance_type": "shap", "feature": "a", "importance_value": 1.0},
            {"target": "A", "model": "XGB", "importance_type": "shap", "feature": "c", "importance_value": 2.0},
        ]
    )
    out = deterministic_top_n(df, ["target", "model", "importance_type"], "importance_value", 2)
    assert out["feature"].tolist() == ["c", "a"]
    assert out["rank"].tolist() == [1, 2]


def test_tft_attention_is_not_labelled_as_shap(tmp_path: Path) -> None:
    run_dir = tmp_path / "tft_run"
    reports = run_dir / "reports"
    reports.mkdir(parents=True)
    (reports / "da_target_da_price_attention_interpretation.png").write_bytes(b"fake")

    inventory: list[dict] = []
    warnings_out: list[WarningRow] = []
    _collect_tft(run_dir, tmp_path / "out", inventory, warnings_out)

    assert any(w.warning_type == "tft_not_shap" for w in warnings_out)
    assert not any("shap" in row["artifact_type"].lower() for row in inventory if row["model"] == "TFT")


def test_summary_rows_marks_tft_as_visual_only() -> None:
    top = pd.DataFrame(
        [
            {
                "target": "DA price",
                "model": "XGB",
                "importance_type": "existing_xgb_mean_abs_shap",
                "rank": 1,
                "feature_group": "price history",
            },
            {
                "target": "DA price",
                "model": "RLQR",
                "importance_type": "robust_scaled_p50_coefficient_abs",
                "rank": 1,
                "feature_group": "load",
            },
        ]
    )
    overlap = pd.DataFrame([{"target": "DA price", "jaccard_overlap": 0.0}])
    rows = _summary_rows(top, overlap)
    assert rows[0][1] == "Visual artifact only; see appendix"


def test_latex_table_uses_booktabs(tmp_path: Path) -> None:
    path = _write_latex_table(
        tmp_path / "table.tex",
        ["A", "B"],
        [["x", 1.0]],
        "Caption",
        "tab:test",
    )
    text = path.read_text(encoding="utf-8")
    assert "\\toprule" in text
    assert "\\midrule" in text
    assert "\\bottomrule" in text
