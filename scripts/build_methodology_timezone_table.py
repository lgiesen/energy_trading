"""Generate methodology table documenting time-zone harmonization by data source.

The table is intentionally static and code-audited: it documents how each
ingestion path treats source timestamps before the pipeline merges everything on
the canonical UTC key.

Usage:
    ./.venv/bin/python scripts/build_methodology_timezone_table.py

Outputs by default:
    artifacts/methodology/data_source_time_zones/data_source_time_zones.csv
    artifacts/methodology/data_source_time_zones/data_source_time_zones.md
    artifacts/methodology/data_source_time_zones/data_source_time_zones.tex
    /Users/leori/Desktop/ uni/3 Master IS/25 MA/MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets/figures/3-method/
"""
from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_OUT_DIR = Path("artifacts/methodology/data_source_time_zones")
DEFAULT_EXPORT_DIR = Path(
    "/Users/leori/Desktop/ uni/3 Master IS/25 MA/"
    "MA___Elevated_Energy_Trading__Forecasting_Expected_Profit_From_Participating_in_the_Day_Ahead_and_Balancing_Markets/"
    "figures/3-method"
)


@dataclass(frozen=True)
class TimeZoneRow:
    data_source: str
    raw_timestamp_convention: str
    pipeline_conversion: str
    final_pipeline_time_basis: str
    implementation_reference: str


ROWS: tuple[TimeZoneRow, ...] = (
    TimeZoneRow(
        data_source="ENTSO-E Transparency Platform",
        raw_timestamp_convention=(
            "API responses are handled through entsoe-py as timezone-aware time series; "
            "outage pages may arrive in the ENTSO-E area timezone."
        ),
        pipeline_conversion=(
            "All indices and timestamp columns are localized if needed and converted to UTC."
        ),
        final_pipeline_time_basis="timestamp_utc, hourly UTC",
        implementation_reference=(
            "src/energy_trading/ingestion/fetch_entsoe.py; "
            "scripts/fetch_entsoe_outages.py; scripts/transform_entsoe_outages_hourly.py"
        ),
    ),
    TimeZoneRow(
        data_source="Energy-Charts",
        raw_timestamp_convention="API returns Unix seconds for price observations.",
        pipeline_conversion=(
            "Unix seconds are parsed as UTC; API date boundaries are requested from the "
            "Berlin-local calendar dates covering the UTC analysis window."
        ),
        final_pipeline_time_basis="timestamp, hourly UTC",
        implementation_reference="src/energy_trading/ingestion/fetch_energy_charts.py",
    ),
    TimeZoneRow(
        data_source="Netztransparenz",
        raw_timestamp_convention=(
            "API requests use UTC-style ISO windows; returned time strings are parsed as UTC "
            "in the ingestion code."
        ),
        pipeline_conversion=(
            "Quarter-hour source series are parsed to timestamp_utc and optionally resampled "
            "to hourly UTC aggregates."
        ),
        final_pipeline_time_basis="timestamp_utc, UTC",
        implementation_reference="src/energy_trading/ingestion/fetch_netztransparenz.py",
    ),
    TimeZoneRow(
        data_source="SMARD chart_data JSON",
        raw_timestamp_convention=(
            "Chart-data observations are epoch milliseconds; requested windows are converted "
            "to Europe/Berlin before calculating epoch milliseconds."
        ),
        pipeline_conversion=(
            "Epoch milliseconds are converted to timestamps, truncated to hourly resolution, "
            "then represented as UTC with a CET/Berlin diagnostic timestamp."
        ),
        final_pipeline_time_basis="timestamp_utc, UTC; timestamp_cet for diagnostics",
        implementation_reference="src/energy_trading/ingestion/fetch_smard.py",
    ),
    TimeZoneRow(
        data_source="SMARD market-data CSV export",
        raw_timestamp_convention="CSV start dates are local German market time.",
        pipeline_conversion=(
            "Start dates are localized to Europe/Berlin with DST handling and converted to UTC "
            "before joining installed-capacity columns."
        ),
        final_pipeline_time_basis="timestamp_utc, hourly UTC",
        implementation_reference="src/energy_trading/ingestion/fetch_smard.py",
    ),
    TimeZoneRow(
        data_source="Regelleistung.net",
        raw_timestamp_convention=(
            "Published XLSX/ZIP result files encode delivery dates and product hours in "
            "German local market time."
        ),
        pipeline_conversion=(
            "Delivery dates/product hours are localized to Europe/Berlin with DST handling "
            "and converted to UTC; 15-minute series are aggregated to hourly outputs where needed."
        ),
        final_pipeline_time_basis="timestamp_utc, UTC",
        implementation_reference="src/energy_trading/ingestion/fetch_regelleistung.py",
    ),
    TimeZoneRow(
        data_source="Yahoo Finance",
        raw_timestamp_convention="Daily financial-market dates are parsed as UTC dates.",
        pipeline_conversion=(
            "Daily closes are shifted by one day so close(D) becomes available from D+1 "
            "00:00 UTC, then forward-filled to hourly UTC timestamps."
        ),
        final_pipeline_time_basis="timestamp, hourly UTC",
        implementation_reference="src/energy_trading/ingestion/fetch_yfinance.py",
    ),
    TimeZoneRow(
        data_source="Merged modeling table",
        raw_timestamp_convention="Input sources may expose timestamp, timestamp_utc, or timestamp_cet.",
        pipeline_conversion=(
            "All source tables are normalized to a canonical UTC join key; duplicate/null "
            "timestamps are removed and only timestamp_utc plus timestamp_cet diagnostics are kept."
        ),
        final_pipeline_time_basis="timestamp_utc, UTC; timestamp_cet diagnostic only",
        implementation_reference="src/energy_trading/ingestion/merge_data.py",
    ),
)


def _latex_escape(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def _write_csv(rows: tuple[TimeZoneRow, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_markdown(rows: tuple[TimeZoneRow, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Data source",
        "Raw timestamp convention",
        "Pipeline conversion",
        "Final time basis",
        "Implementation reference",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        vals = [
            row.data_source,
            row.raw_timestamp_convention,
            row.pipeline_conversion,
            row.final_pipeline_time_basis,
            row.implementation_reference,
        ]
        safe_vals = [str(v).replace("|", "\\|").replace("\n", " ") for v in vals]
        lines.append("| " + " | ".join(safe_vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex(rows: tuple[TimeZoneRow, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Time-zone harmonization by data source. All source-specific timestamp conventions are converted to a canonical UTC timestamp before merging and model feature construction.}",
        r"\label{tab:data-source-time-zones}",
        r"\begin{tabularx}{\linewidth}{p{0.18\linewidth} p{0.22\linewidth} p{0.27\linewidth} p{0.16\linewidth} X}",
        r"\toprule",
        r"\textbf{Data source} & \textbf{Raw timestamp convention} & \textbf{Pipeline conversion} & \textbf{Final time basis} & \textbf{Implementation reference} \\",
        r"\midrule",
    ]
    for row in rows:
        vals = [
            row.data_source,
            row.raw_timestamp_convention,
            row.pipeline_conversion,
            row.final_pipeline_time_basis,
            row.implementation_reference,
        ]
        lines.append(" & ".join(_latex_escape(v) for v in vals) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _copy_outputs(source_dir: Path, export_dir: Path) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, export_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate methodology table for data-source time-zone handling.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for CSV/Markdown/LaTeX files.")
    parser.add_argument(
        "--export-dir",
        default=str(DEFAULT_EXPORT_DIR),
        help="Directory to copy generated files to. Defaults to the thesis figures/3-method folder.",
    )
    parser.add_argument("--skip-export", action="store_true", help="Do not copy outputs to --export-dir.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "data_source_time_zones.csv"
    md_path = out_dir / "data_source_time_zones.md"
    tex_path = out_dir / "data_source_time_zones.tex"

    _write_csv(ROWS, csv_path)
    _write_markdown(ROWS, md_path)
    _write_latex(ROWS, tex_path)

    if not args.skip_export and args.export_dir:
        _copy_outputs(out_dir, Path(args.export_dir))

    print(f"[OK] wrote {csv_path}")
    print(f"[OK] wrote {md_path}")
    print(f"[OK] wrote {tex_path}")
    if not args.skip_export and args.export_dir:
        print(f"[OK] exported copies to {Path(args.export_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
