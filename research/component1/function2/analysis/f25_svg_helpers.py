"""Minimal, dependency-free grouped-bar-chart SVG writer for F2.5 figures.

No matplotlib is available in the execution environment used to run this
analysis (confirmed: backend/.venv has numpy/scipy/torch/ultralytics/pytest
but not matplotlib or pandas). Figures are hand-authored SVG instead --
plain, real-data grouped bar charts, legible in any browser, with no
fabricated or interpolated values: every bar is a direct real value from
this stage's own tables, and any missing/skipped value is simply omitted
(labeled, never filled).
"""
from __future__ import annotations

COLORS = {
    "gradcam": "#4C78A8",
    "gradcampp": "#F58518",
    "Simple": "#54A24B",
    "Moderate": "#EECA3B",
    "High": "#B279A2",
    "neutral": "#888888",
}


def grouped_bar_chart(
    title: str,
    group_labels: list[str],
    series: dict[str, list[float | None]],
    series_colors: dict[str, str] | None = None,
    y_label: str = "",
    y_max: float | None = None,
    note: str = "",
    width: int = 760,
    height: int = 440,
) -> str:
    """series: {series_name: [value_per_group (None allowed)]}"""
    series_colors = series_colors or {}
    margin_l, margin_r, margin_t, margin_b = 70, 30, 60, 90
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    all_vals = [v for vals in series.values() for v in vals if v is not None]
    if y_max is None:
        y_max = max(all_vals) * 1.15 if all_vals else 1.0
    if y_max <= 0:
        y_max = 1.0

    n_groups = len(group_labels)
    n_series = len(series)
    group_w = plot_w / max(n_groups, 1)
    bar_w = group_w / (n_series + 1)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')
    svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="16" font-weight="bold" fill="#222">{title}</text>')

    # y axis gridlines + labels
    n_ticks = 5
    for i in range(n_ticks + 1):
        y_val = y_max * i / n_ticks
        y_px = margin_t + plot_h - (y_val / y_max) * plot_h
        svg.append(f'<line x1="{margin_l}" y1="{y_px:.1f}" x2="{width-margin_r}" y2="{y_px:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
        svg.append(f'<text x="{margin_l-8}" y="{y_px+4:.1f}" text-anchor="end" font-size="11" fill="#555">{y_val:.2f}</text>')
    svg.append(f'<text x="18" y="{margin_t+plot_h/2:.1f}" text-anchor="middle" font-size="12" fill="#333" '
               f'transform="rotate(-90 18 {margin_t+plot_h/2:.1f})">{y_label}</text>')

    # bars
    for gi, glabel in enumerate(group_labels):
        gx0 = margin_l + gi * group_w
        for si, (sname, vals) in enumerate(series.items()):
            v = vals[gi] if gi < len(vals) else None
            color = series_colors.get(sname, COLORS["neutral"])
            bx = gx0 + (si + 0.5) * bar_w
            if v is None:
                svg.append(f'<text x="{bx:.1f}" y="{margin_t+plot_h-4}" text-anchor="middle" font-size="10" fill="#aaa">n/a</text>')
                continue
            bh = (v / y_max) * plot_h
            by = margin_t + plot_h - bh
            svg.append(f'<rect x="{bx - bar_w*0.4:.1f}" y="{by:.1f}" width="{bar_w*0.8:.1f}" height="{bh:.1f}" fill="{color}"/>')
            svg.append(f'<text x="{bx:.1f}" y="{by-4:.1f}" text-anchor="middle" font-size="10" fill="#222">{v:.3f}</text>')
        svg.append(f'<text x="{gx0+group_w/2:.1f}" y="{margin_t+plot_h+18}" text-anchor="middle" font-size="12" fill="#222">{glabel}</text>')

    # axis line
    svg.append(f'<line x1="{margin_l}" y1="{margin_t+plot_h}" x2="{width-margin_r}" y2="{margin_t+plot_h}" stroke="#333" stroke-width="1.5"/>')
    svg.append(f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{margin_t+plot_h}" stroke="#333" stroke-width="1.5"/>')

    # legend
    lx = margin_l
    ly = height - 34
    for sname in series:
        color = series_colors.get(sname, COLORS["neutral"])
        svg.append(f'<rect x="{lx}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        svg.append(f'<text x="{lx+16}" y="{ly}" font-size="11" fill="#222">{sname}</text>')
        lx += 18 + 9 * len(sname)

    if note:
        svg.append(f'<text x="{margin_l}" y="{height-10}" font-size="9" fill="#777">{note}</text>')

    svg.append('</svg>')
    return "\n".join(svg)
