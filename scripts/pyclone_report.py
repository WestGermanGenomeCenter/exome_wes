#!/usr/bin/env python3
"""
pyclone6_report.py — Visual interpretation report for PyClone6 clonal clustering.

One plot per page, all pages DIN A4 (8.27 × 11.69 inches).

Usage:
    python pyclone6_report.py \\
        --results  sg070_pyclone6_results.tsv  \\
        [--input   sg070_pyclone6_input.tsv]   \\
        [--sample  sg070]                       \\
        [--output  sg070_pyclone6_report.pdf]

Input files:
    results TSV : mutation_id  sample_id  cluster_id  cellular_prevalence
                  cellular_prevalence_std  cluster_assignment_prob
                  variant_allele_frequency
    input TSV   : mutation_id  sample_id  ref_counts  alt_counts
                  normal_cn  minor_cn  major_cn  tumour_content
                  (optional — enables CN and depth pages)

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
from matplotlib.backends.backend_pdf import PdfPages

# ── DIN A4 ────────────────────────────────────────────────────────────────────
A4 = (8.27, 11.69)

PLOT_TOP    = 0.88
PLOT_BOTTOM = 0.14
PLOT_LEFT   = 0.10
PLOT_RIGHT  = 0.95

# ── palette ────────────────────────────────────────────────────────────────────
_BASE_COLOURS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#64B5CD", "#CCB974", "#E377C2",
    "#7F7F7F", "#BCBD22",
]

CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
CHR_LENGTHS = {
    "chr1":249e6,"chr2":242e6,"chr3":198e6,"chr4":190e6,"chr5":182e6,
    "chr6":171e6,"chr7":159e6,"chr8":145e6,"chr9":138e6,"chr10":134e6,
    "chr11":135e6,"chr12":133e6,"chr13":115e6,"chr14":107e6,"chr15":102e6,
    "chr16":90e6,"chr17":83e6,"chr18":80e6,"chr19":59e6,"chr20":63e6,
    "chr21":48e6,"chr22":51e6,"chrX":156e6,"chrY":58e6,"chrM":17e3,
}

# ── shared helpers ─────────────────────────────────────────────────────────────

def _cluster_order(df):
    med = df.groupby("cluster_id")["cellular_prevalence"].median()
    return med.sort_values(ascending=False).index.tolist()


def _palette(cluster_ids):
    labels = sorted(cluster_ids, key=lambda x: int(x) if str(x).isdigit() else x)
    return {lbl: _BASE_COLOURS[i % len(_BASE_COLOURS)] for i, lbl in enumerate(labels)}


def _clonal_class(ccf):
    if ccf >= 0.75:   return "Clonal"
    if ccf >= 0.35:   return "Subclonal"
    return "Low-CCF"


def _new_page(title, sample_id):
    """Fresh A4 figure with suptitle. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=A4)
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=12, fontweight="bold", y=0.95)
    ax.set_position([PLOT_LEFT, PLOT_BOTTOM,
                     PLOT_RIGHT - PLOT_LEFT, PLOT_TOP - PLOT_BOTTOM])
    return fig, ax


def _new_page_noax(title, sample_id):
    """A4 figure with suptitle, no default axes."""
    fig = plt.figure(figsize=A4)
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=12, fontweight="bold", y=0.95)
    return fig


def _style(ax, title, xlabel, ylabel, title_fs=11):
    ax.set_title(title, fontsize=title_fs, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _caption(fig, text, fontsize=8.0):
    fig.text(PLOT_LEFT, 0.02, text,
             ha="left", va="bottom", fontsize=fontsize,
             fontstyle="italic", color="#444444", wrap=True,
             transform=fig.transFigure)


def _save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)

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
            res["variant_allele_frequency"], errors="coerce")
    else:
        res["variant_allele_frequency"] = np.nan

    res["cluster_id_raw"] = res["cluster_id"]
    res["cluster_id"]     = "C" + res["cluster_id"].astype(str)

    parts        = res["mutation_id"].str.split(":", expand=True)
    res["chrom"] = parts[0] if parts.shape[1] > 0 else pd.NA
    res["pos"]   = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan

    if input_path is not None:
        inp = _safe_read(input_path, "input TSV")
        if inp is not None:
            inp_req = {"mutation_id","ref_counts","alt_counts",
                       "minor_cn","major_cn","tumour_content"}
            missing_inp = inp_req - set(inp.columns)
            if missing_inp:
                print(f"  [WARN] input TSV missing: {missing_inp}", file=sys.stderr)
            else:
                for col in ("ref_counts","alt_counts","minor_cn",
                            "major_cn","tumour_content"):
                    inp[col] = pd.to_numeric(inp[col], errors="coerce")
                inp["depth"] = inp["ref_counts"] + inp["alt_counts"]
                res = res.merge(
                    inp[["mutation_id","ref_counts","alt_counts",
                          "depth","minor_cn","major_cn","tumour_content"]],
                    on="mutation_id", how="left")
    else:
        print("  [INFO] --input not supplied; CN/depth plots skipped", file=sys.stderr)

    return res

# ── pages ──────────────────────────────────────────────────────────────────────

# ── p1: summary table ─────────────────────────────────────────────────────────
def page_summary_table(pdf, df, sample_id, pal):
    order   = _cluster_order(df)
    n_total = len(df)
    k       = df["cluster_id"].nunique()

    stats = (df.groupby("cluster_id")
               .agg(n=("mutation_id","count"),
                    ccf_med=("cellular_prevalence","median"),
                    ccf_std=("cellular_prevalence_std","mean"),
                    prob_mean=("cluster_assignment_prob","mean"))
               .reindex(order))
    stats["pi_pct"] = 100 * stats["n"] / n_total
    stats["class"]  = stats["ccf_med"].apply(_clonal_class)

    tc = df["tumour_content"].dropna().mean() if "tumour_content" in df.columns else None

    fig, ax = _new_page("Summary — PyClone6 Clonal Clustering", sample_id)
    ax.axis("off")

    rows = [
        ["Total mutations", str(n_total)],
        ["Clusters (k)",    str(k)],
        ["Tumour purity",   f"{tc:.1%}" if tc is not None else "N/A"],
    ]
    for cl in order:
        r = stats.loc[cl]
        rows.append([
            f"Cluster {cl}",
            f"n={int(r['n'])} ({r['pi_pct']:.1f}%)   "
            f"CCF={r['ccf_med']:.3f} ± {r['ccf_std']:.3f}   "
            f"conf={r['prob_mean']:.2f}   [{r['class']}]"
        ])

    # plain-language section
    clonal    = [c for c in order if stats.loc[c,"class"] == "Clonal"]
    subclonal = [c for c in order if stats.loc[c,"class"] == "Subclonal"]
    lowccf    = [c for c in order if stats.loc[c,"class"] == "Low-CCF"]
    if clonal:
        rows.append(["► Clonal",    f"{', '.join(clonal)} — CCF ≥ 0.75, founding event"])
    if subclonal:
        rows.append(["► Subclonal", f"{', '.join(subclonal)} — 0.35 ≤ CCF < 0.75"])
    if lowccf:
        rows.append(["► Low-CCF",   f"{', '.join(lowccf)} — CCF < 0.35, rare / artefact?"])

    tbl = ax.table(cellText=rows, colLabels=["Property", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.2, 2.1)
    for j in range(2):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2): tbl[i, j].set_facecolor(shade)
    # colour cluster rows
    for i, cl in enumerate(order, start=4):
        tbl[i, 0].set_facecolor(pal.get(cl, "#CCCCCC"))
        tbl[i, 0].set_text_props(color="white", fontweight="bold")

    _caption(fig,
        "PyClone6 clusters somatic mutations by Cancer Cell Fraction (CCF = fraction of tumour "
        "cells carrying that mutation), correcting for copy number and tumour purity. "
        "CCF ≥ 0.75 → clonal; 0.35–0.75 → subclonal; < 0.35 → low-CCF.")
    _save(pdf, fig)


# ── p2: CCF lollipop ──────────────────────────────────────────────────────────
def page_ccf_lollipop(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    stats = (df.groupby("cluster_id")
               .agg(ccf_med=("cellular_prevalence","median"),
                    ccf_std=("cellular_prevalence_std","mean"))
               .reindex(order))

    fig, ax = _new_page("Median CCF per Cluster  (error = mean σ)", sample_id)
    y    = np.arange(len(order))
    ccfs = stats["ccf_med"].values
    stds = stats["ccf_std"].values
    cols = [pal.get(c,"#888888") for c in order]

    ax.hlines(y, 0, ccfs, color=cols, lw=3, alpha=0.7)
    ax.scatter(ccfs, y, color=cols, s=120, zorder=5)
    ax.errorbar(ccfs, y, xerr=stds, fmt="none",
                ecolor="#555555", capsize=4, lw=1.4, alpha=0.65)
    ax.axvline(1.0, color="#E74C3C", ls="--", lw=1.3, label="CCF = 1.0  (fully clonal)")
    ax.axvline(0.5, color="#F39C12", ls=":",  lw=1.1, label="CCF = 0.5")
    ax.set_yticks(y); ax.set_yticklabels(order, fontsize=10)
    ax.set_xlim(0, 1.22)
    ax.legend(fontsize=9)
    for i, (cl, ccf, std) in enumerate(zip(order, ccfs, stds)):
        ax.text(ccf + 0.02, i, f"{ccf:.3f} ± {std:.3f}",
                va="center", fontsize=9)
    _style(ax, "Median CCF per Cluster  (ordered by CCF, clonal first)",
           "Cancer Cell Fraction (CCF)", "Cluster")
    _caption(fig,
        "Median CCF per cluster, sorted clonal→subclonal. "
        "Error bar = mean per-mutation uncertainty σ from PyClone6. "
        "CCF = fraction of tumour cells carrying that cluster's mutations.")
    _save(pdf, fig)


# ── p3: mutations-per-cluster pie ─────────────────────────────────────────────
def page_pie(pdf, df, sample_id, pal):
    order   = _cluster_order(df)
    n_total = len(df)
    stats   = df.groupby("cluster_id").agg(
        n=("mutation_id","count"),
        ccf_med=("cellular_prevalence","median")).reindex(order)

    fig, ax = _new_page("Mutation Proportion per Cluster", sample_id)
    ax.set_position([0.15, 0.12, 0.70, 0.72])  # square-ish area

    labels = [f"{cl}\nCCF={stats.loc[cl,'ccf_med']:.2f}\n"
              f"n={int(stats.loc[cl,'n'])}" for cl in order]
    sizes  = [stats.loc[cl, "n"] for cl in order]
    cols   = [pal.get(cl, "#888888") for cl in order]

    wedges, texts, autos = ax.pie(
        sizes, labels=labels, colors=cols,
        autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        textprops={"fontsize": 9})
    for at in autos:
        at.set_fontsize(8.5); at.set_color("white"); at.set_fontweight("bold")
    ax.set_title("Mutations per cluster  (π)", fontsize=12,
                 fontweight="bold", pad=10)
    _caption(fig,
        "Each wedge = one PyClone6 cluster. Area proportional to number of mutations. "
        "Labels show cluster ID, median CCF, and mutation count.")
    _save(pdf, fig)


# ── p4+: CCF histogram — one page per cluster ─────────────────────────────────
def pages_ccf_per_cluster(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    bins  = np.linspace(0, 1, 41)
    x     = np.linspace(0, 1, 300)

    for cl in order:
        sub     = df[df["cluster_id"] == cl]
        col     = pal.get(cl, "#888888")
        n_cl    = len(sub)
        ccf_med = sub["cellular_prevalence"].median()
        ccf_std = sub["cellular_prevalence_std"].mean()
        pi_pct  = 100 * n_cl / len(df)

        fig, ax = _new_page(
            f"CCF Distribution — Cluster {cl}  "
            f"(n={n_cl:,}  π={pi_pct:.1f}%  median CCF={ccf_med:.3f})",
            sample_id)

        ax.hist(sub["cellular_prevalence"], bins=bins,
                color=col, alpha=0.78, edgecolor="white")
        if ccf_std > 0:
            y_g   = norm_dist.pdf(x, ccf_med, ccf_std)
            scale = n_cl * 0.025 / max(y_g.max(), 1e-9)
            ax.plot(x, y_g * scale, color=col, lw=2.5, alpha=0.85)
        ax.axvline(ccf_med, color="black", lw=2, ls="--",
                   label=f"Median CCF = {ccf_med:.3f}")
        ax.axvline(1.0, color="#E74C3C", lw=1.2, ls=":",
                   label="CCF = 1.0")
        ax.legend(fontsize=9)
        _style(ax, f"CCF Distribution — Cluster {cl}",
               "Cancer Cell Fraction (CCF)", "Mutations")
        _caption(fig,
            f"CCF distribution for the {n_cl} mutations in cluster {cl}. "
            "CCF is computed directly by PyClone6 (copy-number and purity corrected). "
            "Black dashed = median. Smooth curve = Gaussian envelope (width = mean σ). "
            "Red dotted = CCF 1.0 (all tumour cells carry the mutation).")
        _save(pdf, fig)


# ── p: CCF vs VAF scatter ─────────────────────────────────────────────────────
def page_ccf_vaf_scatter(pdf, df, sample_id, pal):
    if df["variant_allele_frequency"].isna().all():
        return
    order = _cluster_order(df)
    fig, ax = _new_page("CCF vs. VAF per Mutation", sample_id)

    for cl in order:
        sub = df[df["cluster_id"] == cl].dropna(
            subset=["cellular_prevalence", "variant_allele_frequency"])
        ax.scatter(sub["variant_allele_frequency"], sub["cellular_prevalence"],
                   color=pal.get(cl, "#888888"), s=16, alpha=0.55,
                   edgecolors="none", label=cl)
    xr = np.linspace(0, 0.5, 100)
    ax.plot(xr, 2 * xr, color="#AAAAAA", ls="--", lw=1.5,
            label="CCF = 2×VAF\n(diploid, ideal)")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    _style(ax, "CCF vs VAF  (CN correction effect)",
           "Variant Allele Frequency (VAF)", "Cancer Cell Fraction (CCF)")
    _caption(fig,
        "Each dot = one mutation. Points above the dashed line (CCF = 2×VAF) indicate "
        "copy-number amplification inflating the raw VAF relative to the true CCF. "
        "PyClone6 uses the CN state and tumour purity to correct for this.")
    _save(pdf, fig)


# ── p: 2×VAF vs CCF histogram comparison ─────────────────────────────────────
def page_ccf_vaf_hist(pdf, df, sample_id, pal):
    if df["variant_allele_frequency"].isna().all():
        return
    fig, ax = _new_page("All Mutations: 2×VAF vs CCF Distribution", sample_id)

    bins    = np.linspace(0, 1, 41)
    all_vaf = df["variant_allele_frequency"].dropna()
    ax.hist(2 * all_vaf.clip(upper=0.5), bins=bins, color="#AAAAAA",
            alpha=0.60, edgecolor="none", label="2×VAF  (diploid proxy, uncorrected)")
    ax.hist(df["cellular_prevalence"].dropna(), bins=bins,
            color="#4C72B0", alpha=0.65, edgecolor="none",
            label="CCF  (PyClone6, CN-corrected)")
    ax.legend(fontsize=9)
    _style(ax, "2×VAF vs CCF — Distribution Comparison",
           "Value", "Mutation count")
    _caption(fig,
        "Grey = 2×VAF (naive CCF proxy assuming diploid copy-neutral loci). "
        "Blue = CCF as computed by PyClone6 after correcting for copy number and purity. "
        "A visible shift between the two distributions indicates CN-driven corrections.")
    _save(pdf, fig)


# ── p: per-cluster CCF vs 2×VAF comparison bar ────────────────────────────────
def page_ccf_vaf_per_cluster(pdf, df, sample_id, pal):
    if df["variant_allele_frequency"].isna().all():
        return
    order = _cluster_order(df)
    fig, ax = _new_page("Median CCF vs 2×Median VAF per Cluster", sample_id)

    cl_stats = df.groupby("cluster_id").agg(
        ccf_med=("cellular_prevalence","median"),
        vaf_med=("variant_allele_frequency","median")).reindex(order)
    cl_stats["vaf2_med"] = 2 * cl_stats["vaf_med"]

    x = np.arange(len(order)); w = 0.35
    bars1 = ax.bar(x - w/2, cl_stats["ccf_med"], width=w,
                   color=[pal.get(c,"#888") for c in order],
                   alpha=0.88, edgecolor="white", label="Median CCF  (PyClone6)")
    ax.bar(x + w/2, cl_stats["vaf2_med"], width=w,
           color=[pal.get(c,"#888") for c in order],
           alpha=0.38, edgecolor="white", label="2 × Median VAF  (uncorrected)")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                f"{bar.get_height():.2f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
    ax.set_ylim(0, 1.25); ax.legend(fontsize=9)
    _style(ax, "Median CCF vs 2×Median VAF per Cluster",
           "Cluster", "Value")
    _caption(fig,
        "Solid bars = PyClone6 median CCF; faded bars = 2 × median VAF (no CN correction). "
        "Large differences highlight clusters where copy-number correction has a strong effect.")
    _save(pdf, fig)


# ── p: assignment probability histogram ──────────────────────────────────────
def page_assignment_prob_hist(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    fig, ax = _new_page("Assignment Probability Distribution per Cluster", sample_id)

    bins = np.linspace(0, 1, 31)
    for cl in order:
        sub = df[df["cluster_id"] == cl]["cluster_assignment_prob"].dropna()
        ax.hist(sub, bins=bins, color=pal.get(cl, "#888888"),
                alpha=0.62, edgecolor="none", label=cl)
    ax.axvline(0.7, color="black", ls="--", lw=1.5, label="p = 0.70")
    ax.legend(fontsize=9)
    _style(ax, "cluster_assignment_prob per Mutation",
           "Assignment probability", "Mutation count")
    _caption(fig,
        "Distribution of PyClone6 assignment probabilities. "
        "Values close to 1 indicate high confidence; values near 0.5 or lower "
        "suggest the mutation could plausibly belong to multiple clusters.")
    _save(pdf, fig)


# ── p: % high-confidence per cluster bar ─────────────────────────────────────
def page_confidence_bar(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    fig, ax = _new_page("Assignment Confidence ≥ 0.70 per Cluster", sample_id)

    pct = [100 * (df[df["cluster_id"]==cl]["cluster_assignment_prob"]
                  .dropna().ge(0.7).mean()) for cl in order]
    bars = ax.bar(order, pct, color=[pal.get(c,"#888888") for c in order],
                  edgecolor="white", alpha=0.87)
    for bar, p in zip(bars, pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{p:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(70, color="#AAAAAA", ls=":", lw=1.2, label="70% threshold")
    ax.legend(fontsize=9)
    _style(ax, "% Mutations with Assignment Probability ≥ 0.70",
           "Cluster", "% high-confidence mutations")
    _caption(fig,
        "Fraction of mutations assigned with ≥ 70% confidence to their cluster. "
        "Clusters with low fractions are less well-separated from neighbouring clusters.")
    _save(pdf, fig)


# ── p: CCF uncertainty violin ─────────────────────────────────────────────────
def page_uncertainty_violin(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    fig, ax = _new_page("CCF Uncertainty (σ) per Cluster", sample_id)

    data  = [df[df["cluster_id"]==cl]["cellular_prevalence_std"].dropna().values
             for cl in order]
    valid = [(d, cl) for d, cl in zip(data, order) if len(d) > 1]
    if valid:
        vp = ax.violinplot([d for d,_ in valid],
                           positions=range(len(valid)),
                           showmedians=True, showextrema=False)
        for body, (_,cl) in zip(vp["bodies"], valid):
            body.set_facecolor(pal.get(cl,"#888888")); body.set_alpha(0.68)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2.2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    _style(ax, "Per-mutation CCF Uncertainty (σ) per Cluster",
           "Cluster", "cellular_prevalence_std  (σ)")
    _caption(fig,
        "Per-mutation CCF uncertainty σ from PyClone6. Wider violins = more uncertain CCF estimates. "
        "Clusters with many non-diploid CN states or low depth tend to have higher σ.")
    _save(pdf, fig)


# ── p+: per-mutation CCF dot plot — one page per cluster ─────────────────────
def pages_ccf_dotplot(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    for cl in order:
        sub = df[df["cluster_id"]==cl].sort_values("cellular_prevalence").reset_index(drop=True)
        col = pal.get(cl,"#888888")
        n   = len(sub)
        fig, ax = _new_page(f"Per-mutation CCF ± 2σ — Cluster {cl}  (n={n:,})", sample_id)

        y = np.arange(n)
        ax.barh(y, 4 * sub["cellular_prevalence_std"],
                left=sub["cellular_prevalence"].values - 2*sub["cellular_prevalence_std"].values,
                height=0.6, color=col, alpha=0.18, edgecolor="none")
        ax.scatter(sub["cellular_prevalence"], y,
                   color=col, s=10, zorder=4, edgecolors="none")
        med = sub["cellular_prevalence"].median()
        ax.axvline(med, color="black", ls="--", lw=1.8,
                   label=f"Median = {med:.3f}")
        ax.axvline(1.0, color="#E74C3C", ls=":", lw=1.2)
        ax.set_xlim(0, 1.12); ax.set_yticks([])
        ax.legend(fontsize=9)
        _style(ax, f"Per-mutation CCF — Cluster {cl}  (sorted by CCF)",
               "CCF", "Mutations (sorted)")
        _caption(fig,
            f"Each dot = one of the {n} mutations in cluster {cl}, sorted by CCF. "
            "Shaded band = ± 2σ uncertainty interval. Tight bands and a steep curve "
            "indicate a well-defined cluster; wide bands suggest ambiguity.")
        _save(pdf, fig)


# ── p: depth histogram (all clusters overlaid) ───────────────────────────────
def page_depth_hist(pdf, df, sample_id, pal):
    if "depth" not in df.columns or df["depth"].isna().all():
        return
    order = _cluster_order(df)
    fig, ax = _new_page("Read Depth Distribution per Cluster", sample_id)

    p99 = np.percentile(df["depth"].dropna(), 99)
    for cl in order:
        sub = df[df["cluster_id"]==cl]["depth"].dropna()
        if len(sub) == 0: continue
        ax.hist(sub.clip(upper=p99), bins=50, color=pal.get(cl,"#888888"),
                alpha=0.55, edgecolor="none", label=cl)
    ax.axvline(df["depth"].median(), color="black", ls="--", lw=1.8,
               label=f"Overall median = {df['depth'].median():.0f}×")
    ax.legend(fontsize=9)
    _style(ax, "Read Depth per Cluster  (clipped at 99th percentile)",
           "Total read depth (×)", "Mutation count")
    _caption(fig,
        "Read depth distribution per cluster. Depth should be comparable across clusters; "
        "systematic differences may affect CCF accuracy for low-coverage clusters.")
    _save(pdf, fig)


# ── p: depth boxplot ──────────────────────────────────────────────────────────
def page_depth_boxplot(pdf, df, sample_id, pal):
    if "depth" not in df.columns or df["depth"].isna().all():
        return
    order = _cluster_order(df)
    fig, ax = _new_page("Read Depth Boxplot per Cluster", sample_id)

    data = [df[df["cluster_id"]==cl]["depth"].dropna().values for cl in order]
    bp   = ax.boxplot(data, positions=range(len(order)), patch_artist=True,
                      widths=0.5, medianprops=dict(color="black", lw=2.2),
                      flierprops=dict(marker=".", ms=3, alpha=0.4))
    for patch, cl in zip(bp["boxes"], order):
        patch.set_facecolor(pal.get(cl,"#888888")); patch.set_alpha(0.72)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    ax.axhline(df["depth"].median(), color="#AAAAAA", ls=":", lw=1.2,
               label=f"Overall median = {df['depth'].median():.0f}×")
    ax.legend(fontsize=9)
    _style(ax, "Depth Boxplot per Cluster", "Cluster", "Total read depth (×)")
    _caption(fig,
        "Boxplot of read depth per cluster. The grey dotted line is the overall median depth. "
        "Clusters with systematically lower depth may have less reliable CCF estimates.")
    _save(pdf, fig)


# ── p: CN state bar ───────────────────────────────────────────────────────────
def page_cn_bar(pdf, df, sample_id, pal):
    if "minor_cn" not in df.columns or df["minor_cn"].isna().all():
        return
    fig, ax = _new_page("Copy-Number State Frequency (all mutations)", sample_id)

    df2 = df.copy()
    df2["cn_state"] = df2["minor_cn"].astype(str) + "+" + df2["major_cn"].astype(str)
    counts  = df2["cn_state"].value_counts().head(12)
    colours = ["#55A868" if s == "1+1" else "#4C72B0" for s in counts.index]

    bars = ax.bar(counts.index, counts.values, color=colours,
                  edgecolor="white", alpha=0.87)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    _style(ax, "Copy-Number States  (minor + major CN)",
           "CN state  (minor + major)", "Mutation count")
    _caption(fig,
        "Frequency of each copy-number state across all mutations. "
        "Green (1+1) = diploid copy-neutral, the ideal state for direct CCF interpretation. "
        "Non-diploid states require CN correction by PyClone6.")
    _save(pdf, fig)


# ── p: CN state per cluster stacked bar ──────────────────────────────────────
def page_cn_per_cluster(pdf, df, sample_id, pal):
    if "minor_cn" not in df.columns or df["minor_cn"].isna().all():
        return
    order = _cluster_order(df)
    fig, ax = _new_page("Copy-Number State per Cluster (stacked)", sample_id)

    df2 = df.copy()
    df2["cn_state"] = df2["minor_cn"].astype(str) + "+" + df2["major_cn"].astype(str)
    top_states = df2["cn_state"].value_counts().head(6).index.tolist()
    cn_pal     = plt.cm.Set2(np.linspace(0, 1, len(top_states)))
    x          = np.arange(len(order))
    bottom     = np.zeros(len(order))

    for i, state in enumerate(top_states):
        counts = [(df2[(df2["cluster_id"]==cl) &
                       (df2["cn_state"]==state)].shape[0]) for cl in order]
        ax.bar(x, counts, bottom=bottom, label=state,
               color=cn_pal[i], edgecolor="white", linewidth=0.4)
        bottom += np.array(counts, dtype=float)

    ax.set_xticks(x); ax.set_xticklabels(order, fontsize=10)
    ax.legend(fontsize=9, loc="upper right", title="CN state", title_fontsize=8)
    _style(ax, "CN State Breakdown per Cluster  (top 6 states)",
           "Cluster", "Mutation count")
    _caption(fig,
        "Copy-number state breakdown per cluster. "
        "PyClone6 uses CN to convert VAF → CCF, so clusters dominated by non-diploid "
        "CN states depend more heavily on the accuracy of the CN model.")
    _save(pdf, fig)


# ── p: CCF vs depth scatter ───────────────────────────────────────────────────
def page_ccf_vs_depth(pdf, df, sample_id, pal):
    if "depth" not in df.columns or df["depth"].isna().all():
        return
    order = _cluster_order(df)
    fig, ax = _new_page("CCF vs Read Depth  (log scale)", sample_id)

    for cl in order:
        sub = df[df["cluster_id"]==cl].dropna(
            subset=["cellular_prevalence","depth"])
        ax.scatter(sub["depth"], sub["cellular_prevalence"],
                   color=pal.get(cl,"#888888"), s=12, alpha=0.45,
                   edgecolors="none", label=cl)
    ax.set_xscale("log")
    ax.legend(fontsize=9, markerscale=1.5)
    _style(ax, "CCF vs Read Depth  (log scale)",
           "Read depth (×)  [log]", "Cancer Cell Fraction (CCF)")
    _caption(fig,
        "CCF against read depth on a log scale. Low-depth mutations (left) tend to have "
        "higher uncertainty. No strong depth-CCF correlation is expected in well-called data.")
    _save(pdf, fig)


# ── p: CCF vs sigma scatter ───────────────────────────────────────────────────
def page_ccf_vs_sigma(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    fig, ax = _new_page("CCF vs Uncertainty (σ) per Mutation", sample_id)

    for cl in order:
        sub = df[df["cluster_id"]==cl].dropna(
            subset=["cellular_prevalence","cellular_prevalence_std"])
        ax.scatter(sub["cellular_prevalence"], sub["cellular_prevalence_std"],
                   color=pal.get(cl,"#888888"), s=12, alpha=0.45,
                   edgecolors="none", label=cl)
    ax.legend(fontsize=9, markerscale=1.5)
    _style(ax, "CCF vs Uncertainty σ  (per mutation)",
           "Cancer Cell Fraction (CCF)", "cellular_prevalence_std  (σ)")
    _caption(fig,
        "Per-mutation CCF vs uncertainty σ. Boundary effects (CCF near 0 or 1) "
        "sometimes inflate σ due to constraints in the Dirichlet process model.")
    _save(pdf, fig)


# ── p: clonal architecture bar ───────────────────────────────────────────────
def page_architecture_bar(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    stats = df.groupby("cluster_id").agg(
        n=("mutation_id","count"),
        ccf_med=("cellular_prevalence","median"),
        ccf_std=("cellular_prevalence_std","mean")).reindex(order)
    ccfs = stats["ccf_med"].values
    stds = stats["ccf_std"].values
    ns   = stats["n"].values
    cols = [pal.get(c,"#888888") for c in order]

    fig, ax = _new_page("Clonal Architecture — CCF per Cluster", sample_id)
    bars = ax.bar(order, ccfs, color=cols, edgecolor="white", alpha=0.88,
                  yerr=stds, error_kw=dict(ecolor="#555555", capsize=5, lw=1.4))
    ax.axhline(1.0, color="#E74C3C", ls="--", lw=1.3, label="CCF = 1.0  (clonal)")
    ax.axhline(0.5, color="#F39C12", ls=":",  lw=1.1, label="CCF = 0.5")
    for bar, ccf, std, n in zip(bars, ccfs, stds, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.03,
                f"CCF={ccf:.2f}\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.45); ax.legend(fontsize=9)
    _style(ax, "Median CCF per Cluster  (error bars = mean σ)",
           "Cluster", "Cancer Cell Fraction (CCF)")
    _caption(fig,
        "Estimated CCF per cluster with uncertainty. "
        "Clonal clusters (CCF ≈ 1) were present in the founding tumour cell. "
        "Subclonal clusters arose later in a subset of cells.")
    _save(pdf, fig)


# ── p: clonal architecture schematic ─────────────────────────────────────────
def page_architecture_schematic(pdf, df, sample_id, pal):
    order = _cluster_order(df)
    stats = df.groupby("cluster_id").agg(
        n=("mutation_id","count"),
        ccf_med=("cellular_prevalence","median")).reindex(order)
    ccfs = stats["ccf_med"].values
    ns   = stats["n"].values

    fig, ax = _new_page("Clonal Architecture — Tumour Population Schematic", sample_id)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    cx, cy, max_r = 5.0, 5.0, 3.8
    sorted_arc = sorted(zip(order, ccfs, ns), key=lambda x: -x[1])
    n_circ     = len(sorted_arc)
    for i, (cl, ccf, n) in enumerate(sorted_arc):
        r = max_r * ccf
        c = pal.get(cl, "#888888")
        ax.add_patch(plt.Circle((cx, cy), r, color=c, alpha=0.18, zorder=i))
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                 edgecolor=c, lw=2.2, zorder=i+1))
        angle = (2 * np.pi * i / n_circ) - np.pi / 5
        lx = cx + r * 1.10 * np.cos(angle)
        ly = cy + r * 1.10 * np.sin(angle)
        ax.text(lx, ly, f"{cl}\nCCF={ccf:.2f}\nn={n}",
                ha="center", va="center", fontsize=9, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=c, alpha=0.88))
    ax.set_title("Tumour Cell Population  (circle area ∝ CCF)",
                 fontsize=12, fontweight="bold", pad=10)
    _caption(fig,
        "Schematic of tumour cell populations. Circle area ∝ CCF. "
        "Concentric arrangement shows nested clonal structure — larger outer circles "
        "represent more prevalent clones present in more cells.")
    _save(pdf, fig)


# ── p: chromosome absolute stacked ────────────────────────────────────────────
def page_chrom_absolute(pdf, df, sample_id, pal):
    if df["chrom"].isna().all():
        return
    order       = _cluster_order(df)
    present     = [c for c in CHR_ORDER if c in df["chrom"].dropna().values]
    other       = sorted(set(df["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other
    x           = np.arange(len(chrom_order))

    fig, ax = _new_page("Chromosomal Distribution — Absolute Counts", sample_id)
    ax.set_position([0.08, PLOT_BOTTOM, 0.87, PLOT_TOP - PLOT_BOTTOM])

    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (df[df["cluster_id"]==cl]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        ax.bar(x, counts, bottom=bottom, label=cl,
               color=pal.get(cl,"#888888"), edgecolor="white", linewidth=0.3)
        bottom += counts
    ax.set_xticks(x)
    ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7.5)
    ax.legend(fontsize=9, loc="upper right")
    _style(ax, "Mutations per Chromosome  (stacked by cluster)",
           "Chromosome", "Count")
    _caption(fig,
        "Absolute mutation counts per chromosome, stacked by cluster. "
        "Each chromosome's total bar height = total mutations on that chromosome.")
    _save(pdf, fig)


# ── p: chromosome 100% stacked ────────────────────────────────────────────────
def page_chrom_fraction(pdf, df, sample_id, pal):
    if df["chrom"].isna().all():
        return
    order       = _cluster_order(df)
    present     = [c for c in CHR_ORDER if c in df["chrom"].dropna().values]
    other       = sorted(set(df["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other
    x           = np.arange(len(chrom_order))
    totals      = df.groupby("chrom").size().reindex(chrom_order, fill_value=0).values.astype(float)

    fig, ax = _new_page("Chromosomal Distribution — Cluster Composition (100% stacked)", sample_id)
    ax.set_position([0.08, PLOT_BOTTOM, 0.87, PLOT_TOP - PLOT_BOTTOM])

    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (df[df["cluster_id"]==cl]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        fracs  = np.where(totals > 0, counts / totals, 0)
        ax.bar(x, fracs, bottom=bottom, label=cl,
               color=pal.get(cl,"#888888"), edgecolor="white", linewidth=0.3)
        bottom += fracs
    ax.set_xticks(x)
    ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    _style(ax, "Cluster Composition per Chromosome  (100% stacked)",
           "Chromosome", "Fraction")
    _caption(fig,
        "Fraction of mutations on each chromosome belonging to each cluster. "
        "Uniform composition across chromosomes supports a genuine clonal signal. "
        "Chromosome-specific enrichment may indicate residual CN effects.")
    _save(pdf, fig)


# ── p: genome-wide CCF landscape ─────────────────────────────────────────────
def page_genome_ccf(pdf, df, sample_id, pal):
    if df["chrom"].isna().all() or df["pos"].isna().all():
        return
    order   = _cluster_order(df)
    present = [c for c in CHR_ORDER if c in df["chrom"].dropna().values]
    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    df2      = df[df["chrom"].isin(present)].copy()
    df2["gx"] = df2["chrom"].map(offsets).fillna(0) + df2["pos"].fillna(0)

    fig, ax = _new_page("Genome-wide CCF Landscape", sample_id)
    for cl in order:
        sub = df2[df2["cluster_id"]==cl]
        ax.scatter(sub["gx"], sub["cellular_prevalence"],
                   color=pal.get(cl,"#888888"), s=7, alpha=0.5,
                   edgecolors="none", label=cl)
        ax.errorbar(sub["gx"], sub["cellular_prevalence"],
                    yerr=2*sub["cellular_prevalence_std"],
                    fmt="none", ecolor=pal.get(cl,"#888888"),
                    alpha=0.07, lw=0.5)
    ax.axhline(1.0, color="#E74C3C", lw=0.9, ls=":", alpha=0.6)
    for ch in present:
        ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        ax.text(mid, -0.07, ch.replace("chr",""),
                ha="center", fontsize=6, color="#555555",
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0, cum); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, loc="upper right", ncol=2, markerscale=1.8)
    _style(ax, "CCF across the Genome  (coloured by cluster)",
           "Genomic position (hg38)", "Cancer Cell Fraction (CCF)")
    _caption(fig,
        "Genome-wide CCF per mutation, coloured by cluster. "
        "Faint vertical bars = ± 2σ uncertainty. Red dotted = CCF 1.0. "
        "Regional CCF deviations from the cluster median may indicate unmodelled CN events.")
    _save(pdf, fig)


# ── p: genome-wide depth landscape ────────────────────────────────────────────
def page_genome_depth(pdf, df, sample_id, pal):
    if "depth" not in df.columns or df["depth"].isna().all():
        return
    if df["chrom"].isna().all() or df["pos"].isna().all():
        return
    order   = _cluster_order(df)
    present = [c for c in CHR_ORDER if c in df["chrom"].dropna().values]
    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    df2       = df[df["chrom"].isin(present)].copy()
    df2["gx"] = df2["chrom"].map(offsets).fillna(0) + df2["pos"].fillna(0)

    fig, ax = _new_page("Genome-wide Read Depth Landscape", sample_id)
    for cl in order:
        sub = df2[df2["cluster_id"]==cl]
        ax.scatter(sub["gx"], sub["depth"],
                   color=pal.get(cl,"#888888"), s=6, alpha=0.45,
                   edgecolors="none", label=cl)
    med = df2["depth"].median()
    ax.axhline(med, color="black", ls="--", lw=1.5,
               label=f"Median = {med:.0f}×")
    for ch in present:
        ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        ax.text(mid, -0.04, ch.replace("chr",""),
                ha="center", fontsize=6, color="#555555",
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0, cum)
    ax.legend(fontsize=8, loc="upper right", ncol=2, markerscale=1.8)
    _style(ax, "Read Depth across the Genome  (coloured by cluster)",
           "Genomic position (hg38)", "Read depth (×)")
    _caption(fig,
        "Read depth per mutation across the genome, coloured by cluster assignment. "
        "Depth should be approximately uniform. Valleys may indicate low-mappability regions.")
    _save(pdf, fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PyClone6 report — one plot per DIN A4 page."
    )
    parser.add_argument("--results", "-r", required=True,
                        help="<sample>_pyclone6_results.tsv")
    parser.add_argument("--input",   "-i", default=None,
                        help="<sample>_pyclone6_input.tsv")
    parser.add_argument("--sample",  "-s", default="sample")
    parser.add_argument("--output",  "-o", default=None)
    args = parser.parse_args()

    sid     = args.sample
    out_pdf = Path(args.output) if args.output \
              else Path(f"{sid}_pyclone6_report.pdf")

    print(f"[pyclone6_report] Sample: {sid}")
    df = load_data(args.results, args.input)

    n_cl = df["cluster_id"].nunique()
    print(f"[pyclone6_report] {len(df):,} mutations, {n_cl} clusters: "
          f"{sorted(df['cluster_id'].unique())}")

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
        info = pdf.infodict()
        info["Title"]  = f"PyClone6 Report — {sid}"
        info["Author"] = "pyclone6_report.py"

        page_summary_table(pdf, df, sid, pal)         # summary table
        page_ccf_lollipop(pdf, df, sid, pal)          # CCF lollipop
        page_pie(pdf, df, sid, pal)                   # mutation-count pie
        pages_ccf_per_cluster(pdf, df, sid, pal)      # CCF hist × cluster
        page_ccf_vaf_scatter(pdf, df, sid, pal)       # CCF vs VAF scatter
        page_ccf_vaf_hist(pdf, df, sid, pal)          # 2×VAF vs CCF hist
        page_ccf_vaf_per_cluster(pdf, df, sid, pal)   # per-cluster bar
        page_assignment_prob_hist(pdf, df, sid, pal)  # assignment prob hist
        page_confidence_bar(pdf, df, sid, pal)        # % high-conf bar
        page_uncertainty_violin(pdf, df, sid, pal)    # σ violin
        pages_ccf_dotplot(pdf, df, sid, pal)          # CCF dotplot × cluster
        page_depth_hist(pdf, df, sid, pal)            # depth hist
        page_depth_boxplot(pdf, df, sid, pal)         # depth boxplot
        page_cn_bar(pdf, df, sid, pal)                # CN state bar
        page_cn_per_cluster(pdf, df, sid, pal)        # CN per cluster
        page_ccf_vs_depth(pdf, df, sid, pal)          # CCF vs depth
        page_ccf_vs_sigma(pdf, df, sid, pal)          # CCF vs sigma
        page_architecture_bar(pdf, df, sid, pal)      # architecture bar
        page_architecture_schematic(pdf, df, sid, pal)# architecture schematic
        page_chrom_absolute(pdf, df, sid, pal)        # chrom absolute
        page_chrom_fraction(pdf, df, sid, pal)        # chrom 100%
        page_genome_ccf(pdf, df, sid, pal)            # genome CCF
        page_genome_depth(pdf, df, sid, pal)          # genome depth

    print(f"[pyclone6_report] Done — {out_pdf}")


if __name__ == "__main__":
    main()