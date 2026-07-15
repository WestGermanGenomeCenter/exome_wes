#!/usr/bin/env python3
"""
pyclone6_report.py — Visual interpretation report for PyClone6 clonal clustering.

Usage:
    python pyclone6_report.py \\
        --results  sg070_pyclone6_results.tsv  \\
        [--input   sg070_pyclone6_input.tsv]   \\
        [--sample  sg070]                       \\
        [--output  sg070_pyclone6_report.pdf]

Input files:
    results TSV  : mutation_id  sample_id  cluster_id  cellular_prevalence
                   cellular_prevalence_std  cluster_assignment_prob
                   variant_allele_frequency
    input TSV    : mutation_id  sample_id  ref_counts  alt_counts
                   normal_cn  minor_cn  major_cn  tumour_content
                   (optional but enables CN and depth pages)

Key distinction from VIBER:
    PyClone6 reports cellular_prevalence = CCF directly (CN-corrected).
    VAF is also available for comparison.

Conda deps:
    conda install -c conda-forge matplotlib numpy pandas scipy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as norm_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

# ── colour palette ─────────────────────────────────────────────────────────────
_BASE_COLOURS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#64B5CD", "#CCB974", "#E377C2",
    "#7F7F7F", "#BCBD22",
]

CHR_ORDER  = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
CHR_LENGTHS = {
    "chr1":249e6,"chr2":242e6,"chr3":198e6,"chr4":190e6,"chr5":182e6,
    "chr6":171e6,"chr7":159e6,"chr8":145e6,"chr9":138e6,"chr10":134e6,
    "chr11":135e6,"chr12":133e6,"chr13":115e6,"chr14":107e6,"chr15":102e6,
    "chr16":90e6,"chr17":83e6,"chr18":80e6,"chr19":59e6,"chr20":63e6,
    "chr21":48e6,"chr22":51e6,"chrX":156e6,"chrY":58e6,"chrM":17e3,
}

# ── helpers ────────────────────────────────────────────────────────────────────

def _cluster_order(df):
    """Sort cluster IDs by median CCF descending (clonal first)."""
    med = df.groupby("cluster_id")["cellular_prevalence"].median()
    return med.sort_values(ascending=False).index.tolist()


def _palette(cluster_ids):
    labels = sorted(cluster_ids, key=lambda x: int(x) if str(x).isdigit() else x)
    return {lbl: _BASE_COLOURS[i % len(_BASE_COLOURS)] for i, lbl in enumerate(labels)}


def _style(ax, title, xlabel, ylabel, title_fs=11):
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _caption(fig, text, y=0.04, fontsize=8.5):
    fig.text(0.04, y, text, ha="left", va="bottom", fontsize=fontsize,
             fontstyle="italic", color="#444444", wrap=True,
             transform=fig.transFigure)


def _header(fig, title, sample_id):
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=13, fontweight="bold", y=0.98)


def _clonal_class(ccf):
    if ccf >= 0.75:
        return "Clonal"
    elif ccf >= 0.35:
        return "Subclonal"
    else:
        return "Low-CCF"

# ── data loading ───────────────────────────────────────────────────────────────

def _safe_read(path, label):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] {label} not found: {p}", file=sys.stderr)
        return None
    try:
        df = pd.read_csv(p, sep="\t")
        if df.empty:
            print(f"  [WARN] {label} is empty", file=sys.stderr)
            return None
        return df
    except Exception as e:
        print(f"  [WARN] Could not read {label}: {e}", file=sys.stderr)
        return None


def load_data(results_path, input_path=None):
    res = _safe_read(results_path, "results TSV")
    if res is None:
        sys.exit("ERROR: results TSV is required.")

    # ── validate results ───────────────────────────────────────────────────
    required = {"mutation_id", "cluster_id", "cellular_prevalence",
                "cellular_prevalence_std", "cluster_assignment_prob"}
    missing = required - set(res.columns)
    if missing:
        sys.exit(f"ERROR: results TSV missing columns: {missing}")

    for col in ("cellular_prevalence", "cellular_prevalence_std",
                "cluster_assignment_prob"):
        res[col] = pd.to_numeric(res[col], errors="coerce")

    if "variant_allele_frequency" in res.columns:
        res["variant_allele_frequency"] = pd.to_numeric(
            res["variant_allele_frequency"], errors="coerce"
        )
    else:
        res["variant_allele_frequency"] = np.nan

    # cluster_id may be int or string; normalise to string "C<n>" for display
    res["cluster_id_raw"] = res["cluster_id"]
    res["cluster_id"] = "C" + res["cluster_id"].astype(str)

    # parse chrom / pos from mutation_id
    parts = res["mutation_id"].str.split(":", expand=True)
    res["chrom"] = parts[0] if parts.shape[1] > 0 else pd.NA
    res["pos"]   = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan

    # ── optionally join input TSV ──────────────────────────────────────────
    inp = None
    if input_path is not None:
        inp = _safe_read(input_path, "input TSV")
        if inp is not None:
            inp_req = {"mutation_id", "ref_counts", "alt_counts",
                       "minor_cn", "major_cn", "tumour_content"}
            missing_inp = inp_req - set(inp.columns)
            if missing_inp:
                print(f"  [WARN] input TSV missing columns: {missing_inp} — "
                      f"some plots will be skipped", file=sys.stderr)
            else:
                for col in ("ref_counts", "alt_counts", "minor_cn",
                            "major_cn", "tumour_content"):
                    inp[col] = pd.to_numeric(inp[col], errors="coerce")
                inp["depth"] = inp["ref_counts"] + inp["alt_counts"]
                res = res.merge(
                    inp[["mutation_id", "ref_counts", "alt_counts",
                          "depth", "minor_cn", "major_cn", "tumour_content"]],
                    on="mutation_id", how="left"
                )
    else:
        print("  [INFO] --input not supplied; CN and depth plots will be skipped",
              file=sys.stderr)

    return res, inp

# ── pages ──────────────────────────────────────────────────────────────────────

def page_summary(pdf, df, sample_id, pal):
    """Page 1 — overview table, CCF pie, plain-language interpretation."""
    order   = _cluster_order(df)
    n_total = len(df)
    k       = df["cluster_id"].nunique()

    # per-cluster stats
    stats = (df.groupby("cluster_id")
               .agg(
                   n=("mutation_id", "count"),
                   ccf_median=("cellular_prevalence", "median"),
                   ccf_mean=("cellular_prevalence", "mean"),
                   ccf_std=("cellular_prevalence_std", "mean"),
                   prob_mean=("cluster_assignment_prob", "mean"),
               )
               .reindex(order))
    stats["pi_pct"] = 100 * stats["n"] / n_total
    stats["class"]  = stats["ccf_median"].apply(_clonal_class)

    fig = plt.figure(figsize=(13, 9))
    _header(fig, "PyClone6 — Summary of Clonal Clustering", sample_id)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.91, bottom=0.10, left=0.05, right=0.97,
                           hspace=0.50, wspace=0.40)

    # ── top-left: summary table ────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0, 0])
    ax_t.axis("off")
    rows = []
    for cl in order:
        r = stats.loc[cl]
        rows.append([
            cl,
            f"{int(r['n'])}  ({r['pi_pct']:.1f}%)",
            f"{r['ccf_median']:.3f}",
            f"± {r['ccf_std']:.3f}",
            f"{r['prob_mean']:.2f}",
            r["class"],
        ])
    col_labels = ["Cluster", "Mutations (π)", "Median CCF", "Mean σ", "Mean conf.", "Class"]
    tbl = ax_t.table(cellText=rows, colLabels=col_labels,
                     loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.15, 1.9)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, cl in enumerate(order, start=1):
        tbl[i, 0].set_facecolor(pal.get(cl, "#CCCCCC"))
        tbl[i, 0].set_text_props(color="white", fontweight="bold")
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(1, len(col_labels)):
            tbl[i, j].set_facecolor(shade)
    ax_t.set_title("Cluster Overview", fontsize=10, fontweight="bold", pad=6)

    # ── top-right: CCF pie ─────────────────────────────────────────────────
    ax_p = fig.add_subplot(gs[0, 1])
    pie_labels = [f"{cl}\nCCF={stats.loc[cl,'ccf_median']:.2f}" for cl in order]
    pie_sizes  = [stats.loc[cl, "n"] for cl in order]
    pie_cols   = [pal.get(cl, "#888888") for cl in order]
    wedges, texts, autos = ax_p.pie(
        pie_sizes, labels=pie_labels, colors=pie_cols,
        autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        textprops={"fontsize": 8.5},
    )
    for at in autos:
        at.set_fontsize(7.5); at.set_color("white"); at.set_fontweight("bold")
    ax_p.set_title("Mutations per cluster", fontsize=10, fontweight="bold", pad=6)

    # ── bottom-left: CCF lollipop ──────────────────────────────────────────
    ax_l = fig.add_subplot(gs[1, 0])
    y      = np.arange(len(order))
    ccfs   = [stats.loc[cl, "ccf_median"] for cl in order]
    stds   = [stats.loc[cl, "ccf_std"]    for cl in order]
    cols   = [pal.get(cl, "#888888")      for cl in order]
    ax_l.hlines(y, 0, ccfs, color=cols, lw=2.5, alpha=0.7)
    ax_l.scatter(ccfs, y, color=cols, s=90, zorder=5)
    ax_l.errorbar(ccfs, y, xerr=stds, fmt="none",
                  ecolor="#555555", capsize=3, lw=1.2, alpha=0.6)
    ax_l.axvline(1.0, color="#E74C3C", ls="--", lw=1.2, label="CCF = 1.0")
    ax_l.axvline(0.5, color="#F39C12", ls=":",  lw=1.0, label="CCF = 0.5")
    ax_l.set_yticks(y); ax_l.set_yticklabels(order, fontsize=9)
    ax_l.set_xlim(0, 1.18)
    ax_l.legend(fontsize=8)
    for i, (cl, ccf, std) in enumerate(zip(order, ccfs, stds)):
        ax_l.text(ccf + 0.02, i, f"{ccf:.3f} ± {std:.3f}",
                  va="center", fontsize=8)
    _style(ax_l, "Median CCF per Cluster  (error = mean σ)",
           "Cancer Cell Fraction (CCF)", "Cluster")

    # ── bottom-right: plain-language blurb ────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.axis("off")
    clonal    = [cl for cl in order if stats.loc[cl,"class"] == "Clonal"]
    subclonal = [cl for cl in order if stats.loc[cl,"class"] == "Subclonal"]
    lowccf    = [cl for cl in order if stats.loc[cl,"class"] == "Low-CCF"]

    tc = df["tumour_content"].dropna().mean() if "tumour_content" in df.columns else None
    tc_str = f"  Tumour purity: {tc:.1%}." if tc is not None else ""

    lines = [
        f"PyClone6 fitted {k} cluster(s) to {n_total:,} somatic mutations.",
        f"Unlike VAF-based tools, PyClone6 computes CCF directly,",
        f"correcting for copy number and tumour purity.{tc_str}\n",
        "CCF = fraction of tumour cells carrying each mutation group.\n",
    ]
    if clonal:
        lines.append(f"► CLONAL    {', '.join(clonal)}  (CCF ≥ 0.75) — founding clone.")
    if subclonal:
        lines.append(f"► SUBCLONAL {', '.join(subclonal)}  (0.35 ≤ CCF < 0.75).")
    if lowccf:
        lines.append(f"► LOW-CCF   {', '.join(lowccf)}  (CCF < 0.35) — rare / artefact?")

    ax_b.text(0.03, 0.97, "\n".join(lines), transform=ax_b.transAxes,
              ha="left", va="top", fontsize=9, color="#222222",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB",
                        edgecolor="#AED6F1", alpha=0.85), wrap=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_ccf_distributions(pdf, df, sample_id, pal):
    """Page 2 — CCF distribution per cluster (the primary PyClone6 output)."""
    order = _cluster_order(df)
    k     = len(order)
    ncols = min(k, 3)
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows + 1.0),
                             squeeze=False)
    _header(fig, "CCF Distribution per Cluster  (PyClone6 output)", sample_id)
    fig.subplots_adjust(hspace=0.65, wspace=0.38, top=0.91, bottom=0.13)

    bins = np.linspace(0, 1, 41)
    x    = np.linspace(0, 1, 300)

    for idx, cl in enumerate(order):
        row, col = divmod(idx, ncols)
        ax   = axes[row][col]
        sub  = df[df["cluster_id"] == cl]
        col_ = pal.get(cl, "#888888")
        n_cl = len(sub)
        ccf_med = sub["cellular_prevalence"].median()
        ccf_std = sub["cellular_prevalence_std"].mean()
        pi_pct  = 100 * n_cl / len(df)

        ax.hist(sub["cellular_prevalence"], bins=bins,
                color=col_, alpha=0.75, edgecolor="white", zorder=2)

        # Gaussian envelope using mean σ
        if ccf_std > 0:
            y_g = norm_dist.pdf(x, ccf_med, ccf_std)
            scale = n_cl * 0.025 / max(y_g.max(), 1e-9)
            ax.plot(x, y_g * scale, color=col_, lw=2, alpha=0.85, zorder=5)

        ax.axvline(ccf_med, color="black", lw=1.8, ls="--",
                   label=f"Median CCF = {ccf_med:.3f}")
        ax.axvline(1.0, color="#E74C3C", lw=1.0, ls=":",  label="CCF = 1.0")
        ax.legend(fontsize=7.5, loc="upper left")
        _style(ax, f"{cl}   n={n_cl:,}   π={pi_pct:.1f}%",
               "Cancer Cell Fraction (CCF)", "Mutations", title_fs=9.5)

    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "CCF (Cancer Cell Fraction) = proportion of tumour cells carrying each mutation, "
        "as computed directly by PyClone6 (CN-corrected, purity-corrected). "
        "Black dashed = median CCF. Smooth curve = Gaussian centred on the median, "
        "width = mean per-mutation σ from PyClone6. "
        "Red dotted = CCF 1.0 (all tumour cells)."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_ccf_vs_vaf(pdf, df, sample_id, pal):
    """Page 3 — CCF vs VAF: show what CN correction does."""
    if df["variant_allele_frequency"].isna().all():
        return

    order = _cluster_order(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    _header(fig, "CCF vs. VAF: Effect of Copy-Number Correction", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.38)

    # ── left: scatter CCF vs VAF per mutation ─────────────────────────────
    ax = axes[0]
    for cl in order:
        sub = df[df["cluster_id"] == cl].dropna(
            subset=["cellular_prevalence", "variant_allele_frequency"]
        )
        ax.scatter(sub["variant_allele_frequency"], sub["cellular_prevalence"],
                   color=pal.get(cl, "#888888"), s=14, alpha=0.55,
                   edgecolors="none", label=cl)
    # reference line: CCF = 2*VAF (diploid, no purity correction)
    xr = np.linspace(0, 0.5, 100)
    ax.plot(xr, 2 * xr, color="#AAAAAA", ls="--", lw=1.2,
            label="CCF = 2×VAF\n(diploid, ideal)")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=7.5)
    _style(ax, "CCF vs VAF (per mutation)", "VAF", "CCF (PyClone6)")

    # ── middle: histogram CCF (blue) vs 2×VAF (grey) overlaid ────────────
    ax2 = axes[1]
    bins = np.linspace(0, 1, 41)
    all_vaf = df["variant_allele_frequency"].dropna()
    ax2.hist(2 * all_vaf.clip(upper=0.5), bins=bins, color="#AAAAAA",
             alpha=0.55, edgecolor="none", label="2×VAF (diploid proxy)")
    ax2.hist(df["cellular_prevalence"].dropna(), bins=bins,
             color="#4C72B0", alpha=0.60, edgecolor="none", label="CCF (PyClone6)")
    ax2.legend(fontsize=8)
    _style(ax2, "All Mutations: 2×VAF vs CCF", "Value", "Count")

    # ── right: per-cluster CCF median vs 2*VAF median ────────────────────
    ax3 = axes[2]
    cl_stats = df.groupby("cluster_id").agg(
        ccf_med=("cellular_prevalence", "median"),
        vaf_med=("variant_allele_frequency", "median"),
    ).reindex(order)
    cl_stats["vaf2_med"] = 2 * cl_stats["vaf_med"]
    x = np.arange(len(order))
    w = 0.35
    bars1 = ax3.bar(x - w/2, cl_stats["ccf_med"],   width=w,
                    color=[pal.get(c,"#888") for c in order], alpha=0.85,
                    edgecolor="white", label="Median CCF")
    bars2 = ax3.bar(x + w/2, cl_stats["vaf2_med"],  width=w,
                    color=[pal.get(c,"#888") for c in order], alpha=0.40,
                    edgecolor="white", label="2 × Median VAF")
    for bar in bars1:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=7.5, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(order, fontsize=9)
    ax3.set_ylim(0, 1.2)
    ax3.legend(fontsize=8)
    _style(ax3, "Median CCF vs 2×Median VAF per Cluster",
           "Cluster", "Value")

    _caption(fig,
        "PyClone6 corrects VAF for copy-number state and tumour purity to estimate CCF. "
        "Left: scatter of VAF vs CCF; points above the grey dashed line (CCF = 2×VAF) "
        "indicate CN amplification inflating allele fraction. "
        "Middle: overall distribution shift between raw 2×VAF and CN-corrected CCF. "
        "Right: per-cluster comparison — large differences indicate CN-driven corrections."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_confidence(pdf, df, sample_id, pal):
    """Page 4 — cluster assignment probability and CCF uncertainty."""
    order = _cluster_order(df)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    _header(fig, "Assignment Confidence & CCF Uncertainty", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.38)

    # ── left: histogram of assignment prob per cluster ─────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 31)
    for cl in order:
        sub = df[df["cluster_id"] == cl]["cluster_assignment_prob"].dropna()
        ax.hist(sub, bins=bins, color=pal.get(cl, "#888888"),
                alpha=0.60, edgecolor="none", label=cl)
    ax.axvline(0.7, color="black", ls="--", lw=1.2, label="p = 0.70")
    ax.legend(fontsize=7.5)
    _style(ax, "Assignment Probability per Mutation", "cluster_assignment_prob", "Count")

    # ── middle: % high-confidence per cluster ─────────────────────────────
    ax2 = axes[1]
    pct_high = []
    for cl in order:
        sub = df[df["cluster_id"] == cl]["cluster_assignment_prob"].dropna()
        pct_high.append(100 * (sub >= 0.7).mean() if len(sub) > 0 else 0)
    bars = ax2.bar(order, pct_high,
                   color=[pal.get(c,"#888888") for c in order],
                   edgecolor="white", alpha=0.85)
    for bar, pct in zip(bars, pct_high):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{pct:.0f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.axhline(70, color="#AAAAAA", ls=":", lw=1.0)
    _style(ax2, "% Mutations with prob ≥ 0.70",
           "Cluster", "% high-confidence mutations")

    # ── right: CCF uncertainty (σ) distribution per cluster ──────────────
    ax3 = axes[2]
    data_by_cl = [df[df["cluster_id"] == cl]["cellular_prevalence_std"].dropna().values
                  for cl in order]
    valid = [(d, cl) for d, cl in zip(data_by_cl, order) if len(d) > 1]
    if valid:
        vp = ax3.violinplot([d for d, _ in valid],
                            positions=range(len(valid)),
                            showmedians=True, showextrema=False)
        for body, (_, cl) in zip(vp["bodies"], valid):
            body.set_facecolor(pal.get(cl, "#888888")); body.set_alpha(0.65)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
    ax3.set_xticks(range(len(order)))
    ax3.set_xticklabels(order, fontsize=9)
    _style(ax3, "CCF Uncertainty (σ) per Cluster",
           "Cluster", "cellular_prevalence_std (σ)")

    _caption(fig,
        "cluster_assignment_prob: PyClone6's confidence that a mutation belongs to its "
        "assigned cluster (analogous to the max posterior in VIBER). "
        "Left: distribution per cluster; Middle: fraction ≥ 0.70. "
        "Right: per-mutation CCF uncertainty σ — wider violins indicate clusters where "
        "PyClone6 struggled to pin down the CCF precisely."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_ccf_with_uncertainty(pdf, df, sample_id, pal):
    """Page 5 — per-mutation CCF dot plot with ± 2σ error bars, sorted by CCF."""
    order = _cluster_order(df)

    fig, axes = plt.subplots(1, len(order),
                             figsize=(max(4.5 * len(order), 10), 7),
                             squeeze=False)
    _header(fig, "Per-mutation CCF with Uncertainty  (± 2σ)", sample_id)
    fig.subplots_adjust(bottom=0.18, top=0.90, wspace=0.40)

    for ax, cl in zip(axes[0], order):
        sub = df[df["cluster_id"] == cl].sort_values("cellular_prevalence")
        col = pal.get(cl, "#888888")
        y   = np.arange(len(sub))

        # error bars ± 2σ
        ax.barh(y, 4 * sub["cellular_prevalence_std"],
                left=sub["cellular_prevalence"].values - 2 * sub["cellular_prevalence_std"].values,
                height=0.6, color=col, alpha=0.20, edgecolor="none")
        ax.scatter(sub["cellular_prevalence"], y,
                   color=col, s=12, zorder=4, edgecolors="none")

        med = sub["cellular_prevalence"].median()
        ax.axvline(med, color="black", ls="--", lw=1.4,
                   label=f"Median = {med:.3f}")
        ax.axvline(1.0, color="#E74C3C", ls=":", lw=1.0)
        ax.set_xlim(0, 1.1)
        ax.set_yticks([])
        ax.legend(fontsize=7.5, loc="upper left")
        _style(ax, f"{cl}\n(n = {len(sub):,})", "CCF", "Mutations (sorted)", title_fs=9.5)

    _caption(fig,
        "Each dot = one mutation. Mutations sorted by CCF within their cluster. "
        "Shaded band = ± 2σ uncertainty interval from PyClone6. "
        "Tight bands and steep rank curves indicate a well-defined cluster. "
        "Wide, overlapping bands suggest ambiguity — possibly a single cluster "
        "spanning what are actually two evolutionary populations."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_chromosome_distribution(pdf, df, sample_id, pal):
    """Page 6 — mutations per chromosome, stacked by cluster."""
    if df["chrom"].isna().all():
        return

    order       = _cluster_order(df)
    present     = [c for c in CHR_ORDER if c in df["chrom"].dropna().values]
    other       = sorted(set(df["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    _header(fig, "Chromosomal Distribution of Mutations by Cluster", sample_id)
    fig.subplots_adjust(hspace=0.58, bottom=0.12, top=0.91)

    x = np.arange(len(chrom_order))

    # absolute stacked
    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (df[df["cluster_id"] == cl]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        ax1.bar(x, counts, bottom=bottom, label=cl,
                color=pal.get(cl, "#888888"), edgecolor="white", linewidth=0.3)
        bottom += counts
    ax1.set_xticks(x)
    ax1.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=8, loc="upper right")
    _style(ax1, "Mutations per Chromosome (stacked by cluster)", "Chromosome", "Count")

    # 100% stacked
    totals = df.groupby("chrom").size().reindex(chrom_order, fill_value=0).values.astype(float)
    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (df[df["cluster_id"] == cl]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        fracs = np.where(totals > 0, counts / totals, 0)
        ax2.bar(x, fracs, bottom=bottom, label=cl,
                color=pal.get(cl, "#888888"), edgecolor="white", linewidth=0.3)
        bottom += fracs
    ax2.set_xticks(x)
    ax2.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc="upper right")
    _style(ax2, "Cluster Composition per Chromosome (100% stacked)",
           "Chromosome", "Fraction")

    _caption(fig,
        "Top: absolute mutation counts per chromosome, coloured by cluster. "
        "Bottom: fraction of mutations on each chromosome belonging to each cluster. "
        "Uniform composition across chromosomes supports a genuine clonal signal. "
        "Chromosome-specific enrichment may indicate residual CN effects.",
        y=0.03
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_depth_cn(pdf, df, sample_id, pal):
    """Page 7 — read depth and copy-number state distributions (requires input TSV)."""
    needed = {"depth", "minor_cn", "major_cn"}
    if not needed.issubset(df.columns) or df["depth"].isna().all():
        print("  [INFO] Depth/CN data not available — skipping depth/CN page",
              file=sys.stderr)
        return

    order = _cluster_order(df)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _header(fig, "Read Depth and Copy-Number State", sample_id)
    fig.subplots_adjust(hspace=0.45, wspace=0.38, bottom=0.14, top=0.90)

    # ── top-left: depth histogram per cluster ─────────────────────────────
    ax = axes[0][0]
    p99 = np.percentile(df["depth"].dropna(), 99)
    for cl in order:
        sub = df[df["cluster_id"] == cl]["depth"].dropna()
        ax.hist(sub.clip(upper=p99), bins=40, color=pal.get(cl, "#888888"),
                alpha=0.55, edgecolor="none", label=cl)
    ax.axvline(df["depth"].median(), color="black", ls="--", lw=1.2,
               label=f"Overall median = {df['depth'].median():.0f}×")
    ax.legend(fontsize=7.5)
    _style(ax, "Read Depth per Cluster", "Depth (×)", "Count")

    # ── top-right: depth boxplot per cluster ──────────────────────────────
    ax2 = axes[0][1]
    data_d = [df[df["cluster_id"] == cl]["depth"].dropna().values for cl in order]
    bp = ax2.boxplot(data_d, positions=range(len(order)), patch_artist=True,
                     widths=0.5, medianprops=dict(color="black", lw=2),
                     flierprops=dict(marker=".", ms=3, alpha=0.4))
    for patch, cl in zip(bp["boxes"], order):
        patch.set_facecolor(pal.get(cl, "#888888")); patch.set_alpha(0.7)
    ax2.set_xticks(range(len(order))); ax2.set_xticklabels(order, fontsize=9)
    ax2.axhline(df["depth"].median(), color="#AAAAAA", ls=":", lw=1.0)
    _style(ax2, "Depth Boxplot per Cluster", "Cluster", "Depth (×)")

    # ── bottom-left: CN state bar chart ───────────────────────────────────
    ax3 = axes[1][0]
    df["cn_state"] = df["minor_cn"].astype(str) + "+" + df["major_cn"].astype(str)
    cn_counts = df["cn_state"].value_counts().head(10)
    cn_colours = ["#55A868" if s == "1+1" else "#4C72B0" for s in cn_counts.index]
    bars = ax3.bar(cn_counts.index, cn_counts.values,
                   color=cn_colours, edgecolor="white", alpha=0.85)
    for bar, val in zip(bars, cn_counts.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", fontsize=8)
    ax3.tick_params(axis="x", rotation=35, labelsize=8)
    _style(ax3, "Copy-Number States (minor+major CN)", "CN state", "Mutations")

    # ── bottom-right: CN state per cluster (stacked) ──────────────────────
    ax4 = axes[1][1]
    top_states = df["cn_state"].value_counts().head(6).index.tolist()
    cn_pal = plt.cm.Set2(np.linspace(0, 1, len(top_states)))
    x4 = np.arange(len(order))
    bottom4 = np.zeros(len(order))
    for i, state in enumerate(top_states):
        counts4 = [
            (df[(df["cluster_id"] == cl) & (df["cn_state"] == state)].shape[0])
            for cl in order
        ]
        ax4.bar(x4, counts4, bottom=bottom4,
                label=state, color=cn_pal[i], edgecolor="white", linewidth=0.3)
        bottom4 += np.array(counts4, dtype=float)
    ax4.set_xticks(x4); ax4.set_xticklabels(order, fontsize=9)
    ax4.legend(fontsize=7.5, loc="upper right", title="CN state")
    _style(ax4, "CN State per Cluster (stacked)", "Cluster", "Mutations")

    _caption(fig,
        "Top: read depth distribution per cluster. "
        "Depth should be similar across clusters; systematic differences may affect CCF accuracy. "
        "Bottom-left: copy-number state frequency across all mutations (green = diploid 1+1). "
        "Bottom-right: CN state breakdown per cluster — PyClone6 uses CN to correct VAF→CCF, "
        "so clusters dominated by non-diploid CN states rely more heavily on CN model accuracy.",
        y=0.03
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_ccf_depth_scatter(pdf, df, sample_id, pal):
    """Page 8 — CCF vs read depth scatter."""
    has_depth = "depth" in df.columns and not df["depth"].isna().all()
    has_vaf   = not df["variant_allele_frequency"].isna().all()

    if not has_depth and not has_vaf:
        return

    order = _cluster_order(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _header(fig, "CCF vs. Read Depth", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.38)

    # ── left: CCF vs depth ────────────────────────────────────────────────
    ax = axes[0]
    if has_depth:
        for cl in order:
            sub = df[df["cluster_id"] == cl].dropna(
                subset=["cellular_prevalence", "depth"]
            )
            ax.scatter(sub["depth"], sub["cellular_prevalence"],
                       color=pal.get(cl, "#888888"),
                       s=10, alpha=0.45, edgecolors="none", label=cl)
        ax.set_xscale("log")
        ax.legend(fontsize=7.5, markerscale=1.5)
        _style(ax, "CCF vs Depth  (log scale)", "Read depth (×) [log]", "CCF")
    else:
        ax.text(0.5, 0.5, "Depth data not available\n(provide --input TSV)",
                ha="center", va="center", transform=ax.transAxes, color="#888888")
        ax.axis("off")

    # ── right: CCF vs σ (uncertainty as function of CCF) ─────────────────
    ax2 = axes[1]
    for cl in order:
        sub = df[df["cluster_id"] == cl].dropna(
            subset=["cellular_prevalence", "cellular_prevalence_std"]
        )
        ax2.scatter(sub["cellular_prevalence"], sub["cellular_prevalence_std"],
                    color=pal.get(cl, "#888888"),
                    s=10, alpha=0.45, edgecolors="none", label=cl)
    ax2.legend(fontsize=7.5, markerscale=1.5)
    _style(ax2, "CCF vs Uncertainty (σ)", "CCF", "cellular_prevalence_std (σ)")

    _caption(fig,
        "Left: CCF against read depth (log scale). Low-depth mutations (left side) "
        "tend to have higher CCF uncertainty. Right: CCF vs per-mutation σ — "
        "extreme CCF values near 0 or 1 may have higher uncertainty due to "
        "boundary effects in the Dirichlet process model."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_clonal_architecture(pdf, df, sample_id, pal):
    """Page 9 — clonal architecture: CCF bar + concentric schematic."""
    order = _cluster_order(df)

    stats = df.groupby("cluster_id").agg(
        n=("mutation_id", "count"),
        ccf_med=("cellular_prevalence", "median"),
        ccf_std=("cellular_prevalence_std", "mean"),
    ).reindex(order)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    _header(fig, "Inferred Clonal Architecture", sample_id)
    fig.subplots_adjust(bottom=0.18, top=0.89, wspace=0.42)

    cls   = stats.index.tolist()
    ccfs  = stats["ccf_med"].values
    stds  = stats["ccf_std"].values
    ns    = stats["n"].values
    cols  = [pal.get(c, "#888888") for c in cls]

    bars = ax1.bar(cls, ccfs, color=cols, edgecolor="white", alpha=0.88, yerr=stds,
                   error_kw=dict(ecolor="#555555", capsize=4, lw=1.2))
    ax1.axhline(1.0, color="#E74C3C", ls="--", lw=1.2, label="CCF = 1.0 (clonal)")
    ax1.axhline(0.5, color="#F39C12", ls=":",  lw=1.0, label="CCF = 0.5")
    for bar, ccf, std, n in zip(bars, ccfs, stds, ns):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.03,
                 f"CCF={ccf:.2f}\n(n={n})",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylim(0, 1.40)
    ax1.legend(fontsize=8)
    _style(ax1, "Median CCF per Cluster\n(error bars = mean σ)",
           "Cluster", "Cancer Cell Fraction (CCF)")

    # concentric circles
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("Tumour Cell Population Schematic  (area ∝ CCF)",
                  fontsize=10, fontweight="bold", pad=8)
    cx, cy, max_r = 5.0, 5.0, 3.8
    sorted_arc = sorted(zip(cls, ccfs, ns), key=lambda x: -x[1])
    n_circ = len(sorted_arc)
    for i, (cl, ccf, n) in enumerate(sorted_arc):
        r = max_r * ccf
        c = pal.get(cl, "#888888")
        ax2.add_patch(plt.Circle((cx, cy), r, color=c, alpha=0.18, zorder=i))
        ax2.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=c, lw=2.0, zorder=i+1))
        angle = (2 * np.pi * i / n_circ) - np.pi / 5
        lx = cx + r * 1.08 * np.cos(angle)
        ly = cy + r * 1.08 * np.sin(angle)
        ax2.text(lx, ly, f"{cl}\nCCF={ccf:.2f}",
                 ha="center", va="center", fontsize=8, color=c, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=c, alpha=0.85))

    _caption(fig,
        "CCF estimates from PyClone6 (CN-corrected, purity-corrected). "
        "Clusters with CCF ≈ 1 were present in the founding tumour cell (clonal). "
        "Lower-CCF clusters emerged later in a subset of cells (subclonal). "
        "Error bars and σ values reflect PyClone6's per-mutation uncertainty, "
        "not a confidence interval on the cluster centre itself."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_genome_landscape(pdf, df, sample_id, pal):
    """Page 10 — genome-wide CCF landscape."""
    if df["chrom"].isna().all() or df["pos"].isna().all():
        return

    order   = _cluster_order(df)
    present = [c for c in CHR_ORDER if c in df["chrom"].dropna().values]

    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum
        cum += CHR_LENGTHS.get(ch, 100e6)

    df2 = df[df["chrom"].isin(present)].copy()
    df2["gx"] = df2["chrom"].map(offsets).fillna(0) + df2["pos"].fillna(0)

    has_depth = "depth" in df2.columns and not df2["depth"].isna().all()
    nrows = 2 if has_depth else 1
    fig, axes_all = plt.subplots(nrows, 1, figsize=(15, 5 * nrows + 1), sharex=True)
    if nrows == 1:
        axes_all = [axes_all]
    _header(fig, "Genome-wide Mutation Landscape", sample_id)
    fig.subplots_adjust(hspace=0.10, bottom=0.13, top=0.91)

    ax1 = axes_all[0]
    for cl in order:
        sub = df2[df2["cluster_id"] == cl]
        ax1.scatter(sub["gx"], sub["cellular_prevalence"],
                    color=pal.get(cl, "#888888"),
                    s=8, alpha=0.5, edgecolors="none", label=cl)
        # error bands
        ax1.errorbar(sub["gx"], sub["cellular_prevalence"],
                     yerr=2 * sub["cellular_prevalence_std"],
                     fmt="none", ecolor=pal.get(cl, "#888888"),
                     alpha=0.08, lw=0.5)
    ax1.axhline(1.0, color="#E74C3C", lw=0.8, ls=":", alpha=0.6)
    ax1.set_ylim(0, 1.12)
    ax1.legend(fontsize=7.5, markerscale=1.5, loc="upper right", ncol=2)
    _style(ax1, "CCF across the Genome (coloured by cluster)", "", "CCF")

    if has_depth:
        ax2 = axes_all[1]
        ax2.scatter(df2["gx"], df2["depth"],
                    color="#888888", s=5, alpha=0.3, edgecolors="none")
        _style(ax2, "Read Depth across the Genome",
               "Genomic position (hg38)", "Depth (×)")

    # chromosome lines + labels
    for ch in present:
        for ax in axes_all:
            ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        axes_all[-1].text(mid, axes_all[-1].get_ylim()[1] * 0.88,
                          ch.replace("chr", ""), ha="center", fontsize=6, color="#555555")

    axes_all[0].set_xlim(0, cum)
    _caption(fig,
        "Genome-wide CCF landscape. Each dot = one mutation, coloured by cluster. "
        "Faint vertical bars = ± 2σ CCF uncertainty per mutation. "
        "Red dotted line = CCF 1.0. Regions where CCF deviates systematically "
        "from a cluster's median may indicate unmodelled CN events.",
        y=0.03
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyClone6 visual interpretation report."
    )
    parser.add_argument("--results", "-r", required=True,
                        help="<sample>_pyclone6_results.tsv")
    parser.add_argument("--input", "-i", default=None,
                        help="<sample>_pyclone6_input.tsv (enables depth/CN pages)")
    parser.add_argument("--sample", "-s", default="sample",
                        help="Sample ID for plot titles")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PDF (default: <sample>_pyclone6_report.pdf)")
    args = parser.parse_args()

    sample_id = args.sample
    out_pdf   = Path(args.output) if args.output \
                else Path(f"{sample_id}_pyclone6_report.pdf")

    print(f"[pyclone6_report] Sample: {sample_id}")
    df, inp = load_data(args.results, args.input)

    n_cl = df["cluster_id"].nunique()
    print(f"[pyclone6_report] {len(df):,} mutations, {n_cl} clusters: "
          f"{sorted(df['cluster_id'].unique())}")
    has_depth = "depth" in df.columns and not df["depth"].isna().all()
    has_vaf   = not df["variant_allele_frequency"].isna().all()
    has_coords = not df["chrom"].isna().all()
    print(f"[pyclone6_report] depth={has_depth}, vaf={has_vaf}, "
          f"genomic_coords={has_coords}")

    pal = _palette(df["cluster_id"].unique())

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.5,
    })

    print(f"[pyclone6_report] Writing: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        d = pdf.infodict()
        d["Title"]  = f"PyClone6 Report — {sample_id}"
        d["Author"] = "pyclone6_report.py"

        page_summary(pdf, df, sample_id, pal)               # p1
        page_ccf_distributions(pdf, df, sample_id, pal)     # p2
        page_ccf_vs_vaf(pdf, df, sample_id, pal)            # p3
        page_confidence(pdf, df, sample_id, pal)             # p4
        page_ccf_with_uncertainty(pdf, df, sample_id, pal)  # p5
        page_chromosome_distribution(pdf, df, sample_id, pal) # p6
        page_depth_cn(pdf, df, sample_id, pal)              # p7
        page_ccf_depth_scatter(pdf, df, sample_id, pal)     # p8
        page_clonal_architecture(pdf, df, sample_id, pal)   # p9
        page_genome_landscape(pdf, df, sample_id, pal)      # p10

    print(f"[pyclone6_report] Done — {out_pdf}")


if __name__ == "__main__":
    main()