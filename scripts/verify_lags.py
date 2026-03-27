"""Verify lag correctness and basic PiT leakage constraints.

Usage:
    ./.venv/bin/python scripts/verify_lags.py \
        --path data/features/all_data_features.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


LAG_PATTERN = re.compile(r"^(?P<root>.+)_lag_(?P<hours>\d+)h$")


# Raw columns that must not remain in the final feature artifact due to PiT rules.
PIT_FORBIDDEN_RAW_COLS = {
    "afrr_da_price_spread",
    "afrr_neg_da_price_spread",
    "afrr_activated_mw_pos",
    "afrr_activated_mw_neg",
    "afrr_activation_price_vwap_pos",
    "afrr_activation_price_vwap_neg",
    "afrr_activation_offered_mw_pos",
    "afrr_activation_offered_mw_neg",
    "afrr_capacity_awarded_mw_pos",
    "afrr_capacity_awarded_mw_neg",
    "is_activated",
    "system_stress_signal",
    "grid_stress_index",
    "nrv_zscore_24h",
    "nrv_quantile_5",
    "NRV_balance",
    "residual_load_actual",
    "wind_onshore_actual_entsoe",
    "wind_offshore_actual_entsoe",
    "solar_actual_entsoe",
    "unplanned_outages_mw",
}


@dataclass
class CheckResult:
    ok: bool
    message: str
    severity: str = "ok"


def _compare_numeric(a: pd.Series, b: pd.Series, atol: float = 1e-9) -> tuple[bool, int]:
    aa = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bb = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(bb)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return True, 0
    return bool(np.isclose(aa[valid], bb[valid], rtol=0.0, atol=atol, equal_nan=True).all()), n_valid


def _parse_lag_column(col: str) -> tuple[str, int] | None:
    m = LAG_PATTERN.match(col)
    if not m:
        return None
    return m.group("root"), int(m.group("hours"))


def verify_lag_math(df: pd.DataFrame) -> tuple[list[CheckResult], int, int]:
    cols = list(df.columns)
    lag_cols = [c for c in cols if _parse_lag_column(c)]
    results: list[CheckResult] = []
    failures = 0
    warnings = 0
    for lag_col in sorted(lag_cols):
        root, lag_h = _parse_lag_column(lag_col)  # type: ignore[misc]
        expected = None
        ref_msg = ""
        if root in df.columns:
            expected = df[root].shift(lag_h)
            ref_msg = f"{root}.shift({lag_h}) [aus Feature-Frame]"

        if expected is None:
            warnings += 1
            results.append(
                CheckResult(
                    ok=True,
                    severity="warn",
                    message=(
                        f"⚠️ Hinweis zu {lag_col}: Basisspalte `{root}` ist im finalen "
                        "Artefakt nicht enthalten; exakte Lag-Mathematik daher hier nicht direkt prüfbar."
                    ),
                )
            )
            continue

        is_ok, n_valid = _compare_numeric(df[lag_col], expected)

        if n_valid == 0:
            warnings += 1
            results.append(
                CheckResult(
                    ok=True,
                    severity="warn",
                    message=(
                        f"⚠️ Hinweis zu {lag_col}: keine validen Vergleichswerte "
                        f"(Referenz: {ref_msg})."
                    ),
                )
            )
            continue

        if is_ok:
            results.append(
                CheckResult(
                    ok=True,
                    message=(
                        f"✅ Spalte {lag_col} (Lag {lag_h}h) korrekt "
                        f"[Referenz: {ref_msg}, n={n_valid}]"
                    ),
                )
            )
        else:
            failures += 1
            results.append(
                CheckResult(
                    ok=False,
                    message=(
                        f"❌ FEHLER in {lag_col}: stimmt nicht mit {ref_msg} "
                        f"überein (n={n_valid})."
                    ),
                )
            )

    return results, failures, warnings


def verify_leakage_rules(df: pd.DataFrame) -> tuple[list[CheckResult], int]:
    cols = set(df.columns)
    results: list[CheckResult] = []
    failures = 0

    present_forbidden = sorted(c for c in PIT_FORBIDDEN_RAW_COLS if c in cols)
    if not present_forbidden:
        results.append(
            CheckResult(
                ok=True,
                message="✅ Leakage-Test: keine verbotenen ungelaggten PiT-Spalten gefunden.",
            )
        )
    else:
        failures += len(present_forbidden)
        for c in present_forbidden:
            results.append(
                CheckResult(
                    ok=False,
                    message=(
                        f"❌ FEHLER in {c}: ungelaggte PiT-kritische Spalte "
                        "ist im Artefakt vorhanden."
                    ),
                )
            )

    if "afrr_da_price_spread" in cols:
        failures += 1
        results.append(
            CheckResult(
                ok=False,
                message="❌ FEHLER: roher Spread `afrr_da_price_spread` verbleibt im Artefakt.",
            )
        )
    else:
        results.append(
            CheckResult(
                ok=True,
                message="✅ Roher Spread-Check: `afrr_da_price_spread` nicht vorhanden.",
            )
        )

    return results, failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify lag math and PiT leakage constraints.")
    parser.add_argument(
        "--path",
        default="data/features/all_data_features.parquet",
        help="Pfad zur Feature-Parquet-Datei.",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.path)
    print(f"[INFO] Geladene Datei: {args.path}")
    print(f"[INFO] Zeilen: {len(df):,} | Spalten: {len(df.columns):,}")

    lag_results, lag_failures, lag_warnings = verify_lag_math(df)
    leak_results, leak_failures = verify_leakage_rules(df)

    print("\n=== Mathematische Lag-Prüfung ===")
    for r in lag_results:
        print(r.message)

    print("\n=== Leakage-Prüfung (PiT) ===")
    for r in leak_results:
        print(r.message)

    total_failures = lag_failures + leak_failures
    if total_failures == 0:
        print(f"\n[ERGEBNIS] ✅ Alle Prüfungen bestanden. Hinweise: {lag_warnings}")
        sys.exit(0)

    print(
        f"\n[ERGEBNIS] ❌ Prüfungen fehlgeschlagen. "
        f"Fehleranzahl: {total_failures} | Hinweise: {lag_warnings}"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
