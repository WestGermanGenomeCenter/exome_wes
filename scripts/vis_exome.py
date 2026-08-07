#!/usr/bin/env python3
"""
Variant Annotation Interactive Report Generator

Generates an interactive HTML report with multiple visualizations
for annotated variant TSV data using Plotly.

Usage:
    python variant_report.py -i variants.tsv -o report.html
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# =============================================================================
# Configuration
# =============================================================================

TARGET_COLORS = {
    "On target": "#2ca02c",
    "Off target": "#d62728",
}

LAYOUT_DEFAULTS = {
    "template": "plotly_white",
    "autosize": True,
    "margin": dict(l=40, r=40, t=80, b=40),
    "legend_title": "Target",
}

DISTANCE_BINS = [0, 10, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000]

# =============================================================================
# Data Loading & Preprocessing
# =============================================================================

def load_data(input_path: str) -> pd.DataFrame:
    """Load annotated variant TSV and validate required columns."""
    df = pd.read_csv(input_path, sep="\t")
    required = ["chrom", "start", "ref", "alt", "qual", "depth",
                "distance_to_target", "on_target_overlaps", "variant_type",
                "filter", "target_chrom", "target_start", "target_end"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for visualization."""
    df = df.copy()
    df["log_distance"] = np.log10(df["distance_to_target"] + 1)
    df["log_depth"] = np.log10(df["depth"] + 1)
    df["Target"] = df["on_target_overlaps"].map({0: "Off target", 1: "On target"})
    df["Hover"] = (
        "<b>" + df["chrom"] + ":" + df["start"].astype(str) + "</b><br>" +
        df["ref"] + " → " + df["alt"] + "<br>" +
        "Variant: " + df["variant_type"].astype(str) + "<br>" +
        "Filter: " + df["filter"].astype(str) + "<br><br>" +
        "<b>Quality:</b> " + df["qual"].round(2).astype(str) + "<br>" +
        "<b>Depth:</b> " + df["depth"].round(2).astype(str) + "<br>" +
        "<b>Distance:</b> " + df["distance_to_target"].astype(str) + " bp<br><br>" +
        "<b>Nearest target</b><br>" +
        df["target_chrom"].astype(str) + ":" +
        df["target_start"].astype(str) + "-" +
        df["target_end"].astype(str)
    )
    return df


# =============================================================================
# Plot Builders
# =============================================================================

def _targets_present(df: pd.DataFrame) -> list:
    """Return list of target categories actually present in data."""
    present = []
    for t in ["On target", "Off target"]:
        if t in df["Target"].values:
            present.append(t)
    return present


def build_summary_table(df: pd.DataFrame) -> list:
    """QC summary statistics table."""
    on = df[df["Target"] == "On target"]
    off = df[df["Target"] == "Off target"]

    median_on = on["depth"].median() if len(on) > 0 else 0
    median_off = off["depth"].median() if len(off) > 0 else 0
    enrichment = median_on / median_off if median_off > 0 else float("inf")

    stats = [
        ("Total variants", f"{len(df):,}"),
        ("On-target variants", f"{len(on):,}"),
        ("Off-target variants", f"{len(off):,}"),
        ("% On-target", f"{100 * len(on) / len(df):.2f}%" if len(df) > 0 else "N/A"),
        ("", ""),
        ("Median on-target depth", f"{median_on:.1f}" if len(on) > 0 else "N/A"),
        ("Median off-target depth", f"{median_off:.1f}" if len(off) > 0 else "N/A"),
        ("Median depth enrichment", f"{enrichment:.1f}×" if median_off > 0 else "N/A"),
        ("", ""),
        ("Mean off-target distance (bp)",
         f"{off['distance_to_target'].mean():,.0f}" if len(off) > 0 else "N/A"),
        ("Median off-target distance (bp)",
         f"{off['distance_to_target'].median():,.0f}" if len(off) > 0 else "N/A"),
        ("95th percentile off-target distance",
         f"{off['distance_to_target'].quantile(0.95):,.0f}" if len(off) > 0 else "N/A"),
        ("", ""),
        ("Median QUAL", f"{df['qual'].median():.1f}" if len(df) > 0 else "N/A"),
        ("PASS variants", f"{100 * (df['filter'] == 'PASS').mean():.1f}%" if len(df) > 0 else "N/A"),
        ("SNVs", f"{(df['variant_type'] == 'SNV').sum():,}"),
        ("INDELs", f"{(df['variant_type'] != 'SNV').sum():,}"),
        ("", ""),
        ("Max off-target depth", f"{off['depth'].max():.1f}" if len(off) > 0 else "N/A"),
    ]
    stats = [s for s in stats if s[0] != ""]

    table = go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=18, family="Arial Black"),
            align="center",
            height=42,
        ),
        cells=dict(
            values=[[x[0] for x in stats], [x[1] for x in stats]],
            fill_color=[
                ["#ecf0f1" if i % 2 == 0 else "white" for i in range(len(stats))],
                ["white"] * len(stats),
            ],
            align="center",
            font=dict(size=16, family="Arial"),
            height=36,
        ),
        columnwidth=[0.55, 0.45],
    )
    return [table]


def build_3d_scatter(df: pd.DataFrame) -> list:
    """3D scatter: Quality × Depth × Distance."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Scatter3d(x=[], y=[], z=[], mode="markers", name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Scatter3d(
            x=d["qual"], y=d["depth"], z=d["distance_to_target"],
            mode="markers",
            name=target,
            marker=dict(size=4, opacity=0.8, color=TARGET_COLORS[target]),
            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>",
        ))
    return traces


def build_scatter_qual_depth(df: pd.DataFrame) -> list:
    """2D scatter: Quality vs Depth."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Scatter(x=[], y=[], mode="markers", name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Scatter(
            x=d["qual"], y=d["depth"],
            mode="markers",
            marker=dict(color=TARGET_COLORS[target], size=6, opacity=0.75),
            name=target,
            legendgroup=target,
            showlegend=False,
            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>",
        ))
    return traces


def build_violin_depth(df: pd.DataFrame) -> list:
    """Violin plot: Depth distribution."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Violin(y=[], name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Violin(
            y=d["depth"],
            x=[target] * len(d),
            name=target,
            box_visible=True,
            meanline_visible=True,
            line_color=TARGET_COLORS[target],
            showlegend=False,
        ))
    return traces


def build_violin_quality(df: pd.DataFrame) -> list:
    """Violin plot: Quality distribution."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Violin(y=[], name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Violin(
            y=d["qual"],
            x=[target] * len(d),
            name=target,
            box_visible=True,
            meanline_visible=True,
            line_color=TARGET_COLORS[target],
            showlegend=False,
        ))
    return traces


def build_scatter_qual_logdist(df: pd.DataFrame) -> list:
    """Scatter: Quality vs log10(Distance)."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Scatter(x=[], y=[], mode="markers", name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Scatter(
            x=d["log_distance"], y=d["qual"],
            mode="markers",
            marker=dict(color=TARGET_COLORS[target], size=6, opacity=0.75),
            name=target,
            legendgroup=target,
            showlegend=False,
            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>",
        ))
    return traces


def build_capture_falloff(df: pd.DataFrame) -> list:
    """Line plot: Median depth vs distance bins."""
    targets = _targets_present(df)
    if not targets:
        return [go.Scatter(x=[], y=[], mode="lines+markers", name="No data")]

    max_dist = df["distance_to_target"].max()
    bins = DISTANCE_BINS + [max_dist + 1]
    tmp = df.copy()
    tmp["bin"] = pd.cut(tmp["distance_to_target"], bins=bins, include_lowest=True)
    grouped = (
        tmp.groupby(["bin", "Target"], observed=True)["depth"]
        .median()
        .reset_index()
    )

    traces = []
    for target in targets:
        d = grouped[grouped["Target"] == target]
        traces.append(go.Scatter(
            x=[str(x) for x in d["bin"]],
            y=d["depth"],
            mode="lines+markers",
            name=target,
            line=dict(color=TARGET_COLORS[target], width=3),
            marker=dict(size=8),
            showlegend=False,
        ))
    return traces


def build_scatter_logdepth_logdist(df: pd.DataFrame) -> list:
    """Scatter: log10(Depth) vs log10(Distance)."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Scatter(x=[], y=[], mode="markers", name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Scatter(
            x=d["log_distance"], y=d["log_depth"],
            mode="markers",
            marker=dict(color=TARGET_COLORS[target], size=6, opacity=0.7),
            name=target,
            legendgroup=target,
            showlegend=False,
            text=d["Hover"],
            hovertemplate="%{text}<extra></extra>",
        ))
    return traces


def build_histogram_distance(df: pd.DataFrame) -> list:
    """Histogram: log10(Distance) distribution."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Histogram(x=[], name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Histogram(
            x=d["log_distance"],
            name=target,
            marker_color=TARGET_COLORS[target],
            opacity=0.6,
            nbinsx=40,
        ))
    return traces


def build_scatter_qual_by_variant_type(df: pd.DataFrame) -> list:
    """Box plot: Quality by variant type."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Box(x=[], y=[], name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Box(
            x=d["variant_type"],
            y=d["qual"],
            name=target,
            marker_color=TARGET_COLORS[target],
            showlegend=False,
        ))
    return traces


def build_scatter_depth_by_chrom(df: pd.DataFrame) -> list:
    """Box plot: Depth by chromosome."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Box(x=[], y=[], name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Box(
            x=d["chrom"],
            y=d["depth"],
            name=target,
            marker_color=TARGET_COLORS[target],
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False,
        ))
    return traces


def build_heatmap_qual_depth_2d(df: pd.DataFrame) -> list:
    """2D density heatmap: Quality vs Depth."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Histogram2d(x=[], y=[], name="No data")]
    for target in targets:
        d = df[df["Target"] == target]
        traces.append(go.Histogram2d(
            x=d["qual"],
            y=d["depth"],
            name=target,
            colorscale="YlOrRd" if target == "On target" else "Blues",
            showscale=False,
            nbinsx=30,
            nbinsy=30,
        ))
    return traces


def build_ecdf_distance(df: pd.DataFrame) -> list:
    """ECDF: Cumulative distribution of distance."""
    traces = []
    targets = _targets_present(df)
    if not targets:
        return [go.Scatter(x=[], y=[], mode="lines", name="No data")]
    for target in targets:
        d = df[df["Target"] == target].sort_values("distance_to_target")
        n = len(d)
        if n == 0:
            continue
        traces.append(go.Scatter(
            x=d["distance_to_target"],
            y=np.arange(1, n + 1) / n,
            mode="lines",
            name=target,
            line=dict(color=TARGET_COLORS[target], width=2),
            showlegend=False,
        ))
    return traces


# =============================================================================
# Plot Registry
# =============================================================================
# Each entry: (builder_func, title, row, col, colspan, axis_config, spec_type)
# colspan=2 means the plot spans both columns (full width)
# colspan=1 means half-width (side-by-side with another plot)

PLOT_REGISTRY = [
    # Row 1: Table (left) + empty placeholder (right) for balance
    (build_summary_table, "QC Summary Statistics", 1, 1, 1, {}, "domain"),

    # Row 2: Full-width 3D scatter
    (build_3d_scatter, "3D Scatter (Qual × Depth × Dist)", 2, 1, 2,
     {"scene": {"xaxis_title": "Quality", "yaxis_title": "Depth",
                "zaxis_title": "Distance to Target (bp)"}}, "scene"),

    # Row 3: Two side-by-side plots
    (build_scatter_qual_depth, "Quality vs Depth", 3, 1, 1,
     {"xaxis_title": "Quality", "yaxis_title": "Depth"}, "xy"),
    (build_violin_depth, "Depth Distribution", 3, 2, 1,
     {"yaxis_title": "Depth"}, "xy"),

    # Row 4: Two side-by-side plots
    (build_violin_quality, "Quality Distribution", 4, 1, 1,
     {"yaxis_title": "Quality"}, "xy"),
    (build_scatter_qual_logdist, "Quality vs log₁₀(Distance)", 4, 2, 1,
     {"xaxis_title": "log₁₀(Distance + 1)", "yaxis_title": "Quality"}, "xy"),

    # Row 5: Two side-by-side plots
    (build_capture_falloff, "Capture Falloff (Median Depth)", 5, 1, 1,
     {"xaxis_title": "Distance bin", "yaxis_title": "Median depth"}, "xy"),
    (build_scatter_logdepth_logdist, "log₁₀(Depth) vs log₁₀(Distance)", 5, 2, 1,
     {"xaxis_title": "log₁₀(Distance + 1)", "yaxis_title": "log₁₀(Depth + 1)"}, "xy"),

    # Row 6: Two side-by-side plots
    (build_histogram_distance, "Distance Distribution", 6, 1, 1,
     {"xaxis_title": "log₁₀(Distance + 1)", "yaxis_title": "Variant count"}, "xy"),
    (build_scatter_qual_by_variant_type, "Quality by Variant Type", 6, 2, 1,
     {"xaxis_title": "Variant Type", "yaxis_title": "Quality"}, "xy"),

    # Row 7: Two side-by-side plots
    (build_scatter_depth_by_chrom, "Depth by Chromosome", 7, 1, 1,
     {"xaxis_title": "Chromosome", "yaxis_title": "Depth"}, "xy"),
    (build_heatmap_qual_depth_2d, "Quality-Depth Density", 7, 2, 1,
     {"xaxis_title": "Quality", "yaxis_title": "Depth"}, "xy"),

    # Row 8: Full-width ECDF
    (build_ecdf_distance, "Cumulative Fraction of Variants by Distance", 8, 1, 2,
     {"xaxis_title": "Distance to Target (bp)", "yaxis_title": "Cumulative Fraction"}, "xy"),
]


# =============================================================================
# Report Assembly
# =============================================================================

def create_report(df: pd.DataFrame, output_path: str) -> None:
    """Assemble all plots into a single interactive HTML report."""

    max_row = max(p[2] for p in PLOT_REGISTRY)

    # Build specs: 2 columns per row
    specs = []
    for r in range(1, max_row + 1):
        row_plots = [p for p in PLOT_REGISTRY if p[2] == r]
        if not row_plots:
            specs.append([{"type": "xy"}, {"type": "xy"}])
            continue

        # Check if any plot spans full width (colspan=2)
        full_width = any(p[4] == 2 for p in row_plots)
        if full_width:
            plot = [p for p in row_plots if p[4] == 2][0]
            spec_type = plot[6] if len(plot) > 6 else "xy"
            specs.append([{"type": spec_type, "colspan": 2}, None])
        elif len(row_plots) == 1:
            # Single plot on this row (e.g. table at col 1)
            plot = row_plots[0]
            spec_type = plot[6] if len(plot) > 6 else "xy"
            col = plot[3]
            if col == 1:
                specs.append([{"type": spec_type}, {"type": "xy"}])
            else:
                specs.append([{"type": "xy"}, {"type": spec_type}])
        else:
            # Two half-width plots
            left = [p for p in row_plots if p[3] == 1]
            right = [p for p in row_plots if p[3] == 2]
            left_type = left[0][6] if left and len(left[0]) > 6 else "xy"
            right_type = right[0][6] if right and len(right[0]) > 6 else "xy"
            specs.append([{"type": left_type}, {"type": right_type}])

    subplot_titles = []
    for r in range(1, max_row + 1):
        row_plots = [p for p in PLOT_REGISTRY if p[2] == r]
        if not row_plots:
            subplot_titles.append("")
            subplot_titles.append("")
            continue
        full_width = any(p[4] == 2 for p in row_plots)
        if full_width:
            plot = [p for p in row_plots if p[4] == 2][0]
            subplot_titles.append(plot[1])
            subplot_titles.append("")  # placeholder for colspan
        elif len(row_plots) == 1:
            plot = row_plots[0]
            col = plot[3]
            if col == 1:
                subplot_titles.append(plot[1])
                subplot_titles.append("")
            else:
                subplot_titles.append("")
                subplot_titles.append(plot[1])
        else:
            left = [p for p in row_plots if p[3] == 1]
            right = [p for p in row_plots if p[3] == 2]
            subplot_titles.append(left[0][1] if left else "")
            subplot_titles.append(right[0][1] if right else "")

    # Custom row heights: full table, square 3D, normal rest
    row_heights = [1.6] + [2.2] + [0.9] * (max_row - 2)

    fig = make_subplots(
        rows=max_row,
        cols=2,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        row_heights=row_heights,
    )

    # Add traces
    for entry in PLOT_REGISTRY:
        builder, title, row, col, colspan, axis_config = entry[:6]
        traces = builder(df)
        for trace in traces:
            fig.add_trace(trace, row=row, col=col)

        spec_type = entry[6] if len(entry) > 6 else "xy"
        if spec_type == "scene" and "scene" in axis_config:
            fig.update_scenes(**axis_config["scene"])
        elif spec_type != "domain":
            if "xaxis_title" in axis_config:
                fig.update_xaxes(title_text=axis_config["xaxis_title"], row=row, col=col)
            if "yaxis_title" in axis_config:
                fig.update_yaxes(title_text=axis_config["yaxis_title"], row=row, col=col)

    fig.update_layout(
        title="Variant Annotation Report",
        height=400 + max_row * 520,
        **LAYOUT_DEFAULTS,
    )

    pio.write_html(
        fig,
        file=output_path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True},
    )
    print(f"Wrote {output_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive visualization of annotated variant TSV."
    )
    parser.add_argument("-i", "--input", required=True, help="Input annotated TSV")
    parser.add_argument("-o", "--output", required=True, help="Output HTML report")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    df = load_data(args.input)
    df = preprocess(df)

    # Warnings for missing target categories
    targets_present = df["Target"].unique()
    if "On target" not in targets_present:
        warnings.warn("No on-target variants found in dataset — plots will be empty for on-target traces.")
    if "Off target" not in targets_present:
        warnings.warn("No off-target variants found in dataset — plots will be empty for off-target traces.")

    create_report(df, args.output)


if __name__ == "__main__":
    main()
