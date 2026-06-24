from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"


def ensure_dirs() -> None:
    for path in [DATA_DIR / "raw", PROCESSED_DIR, OUTPUTS_DIR, FIGURES_DIR, TABLES_DIR, PROJECT_ROOT / "notebooks"]:
        path.mkdir(parents=True, exist_ok=True)


def format_number(value: float) -> str:
    if pd.isna(value):
        return ""
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}".replace(",", " ")
    if abs(value) >= 100:
        return f"{value:,.2f}".replace(",", " ")
    return f"{value:.6f}".rstrip("0").rstrip(".")


def save_table(df: pd.DataFrame, csv_path: Path, markdown_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=True)
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(to_markdown(df), encoding="utf-8")


def to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    data = data.reset_index()
    columns = [str(c) for c in data.columns]
    rows = []
    for _, row in data.iterrows():
        rows.append([format_number(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v) for v in row])
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    out.extend("| " + " | ".join(cells) + " |" for cells in rows)
    return "\n".join(out) + "\n"


def _svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Arial,Helvetica,sans-serif;fill:#17202a} .muted{fill:#5d6d7e;font-size:12px}",
        ".title{font-size:20px;font-weight:700}.axis{stroke:#aeb6bf;stroke-width:1}.grid{stroke:#e5e8e8;stroke-width:1}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _scale(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return np.full_like(values, (lo + hi) / 2, dtype=float)
    return lo + (values - vmin) * (hi - lo) / (vmax - vmin)


def save_bar_svg(labels: Iterable[str], values: Iterable[float], path: Path, title: str) -> None:
    labels = list(labels)
    values = np.asarray(list(values), dtype=float)
    width, height = 900, 520
    margin_l, margin_r, margin_t, margin_b = 80, 30, 70, 110
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b
    y_min = min(0.0, float(np.nanmin(values)))
    y_max = max(0.0, float(np.nanmax(values)))
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0
    bar_w = plot_w / max(1, len(labels)) * 0.72
    gap = plot_w / max(1, len(labels))

    lines = _svg_header(width, height)
    lines.append(f'<text x="{margin_l}" y="38" class="title">{title}</text>')
    for i in range(6):
        y = margin_t + i * plot_h / 5
        val = y_max - i * (y_max - y_min) / 5
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_l-10}" y="{y+4:.1f}" text-anchor="end" class="muted">{format_number(val)}</text>')
    zero_y = margin_t + (y_max - 0) / (y_max - y_min) * plot_h
    lines.append(f'<line x1="{margin_l}" y1="{zero_y:.1f}" x2="{width-margin_r}" y2="{zero_y:.1f}" class="axis"/>')
    palette = ["#2166ac", "#d6604d", "#4393c3", "#b2182b", "#4d9221", "#9970ab", "#f4a582", "#92c5de"]
    for i, (label, value) in enumerate(zip(labels, values)):
        x = margin_l + i * gap + (gap - bar_w) / 2
        y = margin_t + (y_max - max(value, 0)) / (y_max - y_min) * plot_h
        y0 = margin_t + (y_max - min(value, 0)) / (y_max - y_min) * plot_h
        h = max(1, abs(y0 - y))
        lines.append(f'<rect x="{x:.1f}" y="{min(y,y0):.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{palette[i % len(palette)]}"/>')
        lines.append(f'<text transform="translate({x+bar_w/2:.1f},{height-70}) rotate(-45)" text-anchor="end" class="muted">{label}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_line_svg(x_labels: list[str], series: dict[str, Iterable[float]], path: Path, title: str) -> None:
    width, height = 980, 540
    margin_l, margin_r, margin_t, margin_b = 90, 40, 75, 80
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b
    arrays = {k: np.asarray(list(v), dtype=float) for k, v in series.items()}
    all_values = np.concatenate([v[np.isfinite(v)] for v in arrays.values() if np.isfinite(v).any()])
    y_min, y_max = float(np.nanmin(all_values)), float(np.nanmax(all_values))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    n = max(len(x_labels), 2)

    lines = _svg_header(width, height)
    lines.append(f'<text x="{margin_l}" y="38" class="title">{title}</text>')
    for i in range(6):
        y = margin_t + i * plot_h / 5
        val = y_max - i * (y_max - y_min) / 5
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_l-10}" y="{y+4:.1f}" text-anchor="end" class="muted">{format_number(val)}</text>')
    palette = ["#2166ac", "#b2182b", "#4d9221", "#9970ab", "#f4a582", "#35978f"]
    for idx, (name, values) in enumerate(arrays.items()):
        xs = margin_l + np.arange(len(values)) * plot_w / (n - 1)
        ys = margin_t + (y_max - values) * plot_h / (y_max - y_min)
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y, ok in zip(xs, ys, np.isfinite(values)) if ok)
        color = palette[idx % len(palette)]
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<rect x="{margin_l + idx*150}" y="{height-42}" width="18" height="4" fill="{color}"/>')
        lines.append(f'<text x="{margin_l + idx*150 + 25}" y="{height-36}" class="muted">{name}</text>')
    for i in np.linspace(0, len(x_labels) - 1, min(6, len(x_labels)), dtype=int):
        x = margin_l + i * plot_w / (n - 1)
        lines.append(f'<text x="{x:.1f}" y="{height-58}" text-anchor="middle" class="muted">{x_labels[i]}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_hist_svg(values: Iterable[float], path: Path, title: str, bins: int = 50) -> None:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    counts, edges = np.histogram(values, bins=bins)
    labels = [format_number(edges[i]) for i in range(len(counts))]
    save_bar_svg(labels, counts, path, title)


def save_heatmap_svg(matrix: pd.DataFrame, path: Path, title: str) -> None:
    labels = list(matrix.columns)
    values = matrix.to_numpy(dtype=float)
    n = len(labels)
    cell = max(18, min(42, 720 // max(1, n)))
    margin_l, margin_t = 170, 95
    width = margin_l + n * cell + 40
    height = margin_t + n * cell + 120
    lines = _svg_header(width, height)
    lines.append(f'<text x="28" y="38" class="title">{title}</text>')
    for i, label in enumerate(labels):
        x = margin_l + i * cell + cell / 2
        lines.append(f'<text transform="translate({x:.1f},{margin_t-12}) rotate(-45)" text-anchor="start" class="muted">{label}</text>')
        y = margin_t + i * cell + cell / 2 + 4
        lines.append(f'<text x="{margin_l-8}" y="{y:.1f}" text-anchor="end" class="muted">{label}</text>')
    for r in range(n):
        for c in range(n):
            v = max(-1.0, min(1.0, values[r, c]))
            if v >= 0:
                intensity = int(245 - 150 * v)
                color = f"rgb({intensity},{intensity+5},255)"
            else:
                intensity = int(245 + 150 * v)
                color = f"rgb(255,{intensity+5},{intensity})"
            x = margin_l + c * cell
            y = margin_t + r * cell
            lines.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="#ffffff"/>')
            if cell >= 30:
                lines.append(f'<text x="{x+cell/2:.1f}" y="{y+cell/2+4:.1f}" text-anchor="middle" font-size="10">{v:.2f}</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def save_backtest_svg(
    dates: list[str],
    pnl: Iterable[float],
    var: Iterable[float],
    exceptions: Iterable[bool],
    path: Path,
    title: str,
) -> None:
    width, height = 980, 540
    margin_l, margin_r, margin_t, margin_b = 95, 40, 75, 80
    plot_w, plot_h = width - margin_l - margin_r, height - margin_t - margin_b
    pnl = np.asarray(list(pnl), dtype=float)
    var_line = -np.asarray(list(var), dtype=float)
    exceptions = np.asarray(list(exceptions), dtype=bool)
    values = np.concatenate([pnl[np.isfinite(pnl)], var_line[np.isfinite(var_line)]])
    y_min, y_max = float(np.nanmin(values)), float(np.nanmax(values))
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    n = max(len(dates), 2)
    xs = margin_l + np.arange(len(pnl)) * plot_w / (n - 1)
    y_pnl = margin_t + (y_max - pnl) * plot_h / (y_max - y_min)
    y_var = margin_t + (y_max - var_line) * plot_h / (y_max - y_min)

    lines = _svg_header(width, height)
    lines.append(f'<text x="{margin_l}" y="38" class="title">{title}</text>')
    for i in range(6):
        y = margin_t + i * plot_h / 5
        val = y_max - i * (y_max - y_min) / 5
        lines.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" class="grid"/>')
        lines.append(f'<text x="{margin_l-10}" y="{y+4:.1f}" text-anchor="end" class="muted">{format_number(val)}</text>')
    zero_y = margin_t + (y_max - 0) * plot_h / (y_max - y_min)
    lines.append(f'<line x1="{margin_l}" y1="{zero_y:.1f}" x2="{width-margin_r}" y2="{zero_y:.1f}" class="axis"/>')
    pnl_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, y_pnl))
    var_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, y_var))
    lines.append(f'<polyline points="{pnl_points}" fill="none" stroke="#2166ac" stroke-width="1.8"/>')
    lines.append(f'<polyline points="{var_points}" fill="none" stroke="#b2182b" stroke-width="2.1"/>')
    for x, y, is_exception in zip(xs, y_pnl, exceptions):
        if is_exception:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="#d73027" stroke="#ffffff" stroke-width="1.5"/>')
    for i in np.linspace(0, len(dates) - 1, min(6, len(dates)), dtype=int):
        x = margin_l + i * plot_w / (n - 1)
        lines.append(f'<text x="{x:.1f}" y="{height-58}" text-anchor="middle" class="muted">{dates[i]}</text>')
    lines.append(f'<rect x="{margin_l}" y="{height-42}" width="18" height="4" fill="#2166ac"/>')
    lines.append(f'<text x="{margin_l+25}" y="{height-36}" class="muted">actual P&amp;L</text>')
    lines.append(f'<rect x="{margin_l+160}" y="{height-42}" width="18" height="4" fill="#b2182b"/>')
    lines.append(f'<text x="{margin_l+185}" y="{height-36}" class="muted">-VaR 99%</text>')
    lines.append(f'<circle cx="{margin_l+330}" cy="{height-40}" r="5.5" fill="#d73027" stroke="#ffffff" stroke-width="1.5"/>')
    lines.append(f'<text x="{margin_l+345}" y="{height-36}" class="muted">VaR breach</text>')
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
