#!/usr/bin/env python3
"""
viber_report.py — Visual interpretation report for VIBER clonal clustering.

One plot per page, all pages DIN A4 (8.27 × 11.69 inches).

Usage:
    python viber_report.py \
        --dir    /path/to/viber_export/  \
        [--sample sg070]                 \
        [--output sg070_viber_report.pdf]

    python viber_report.py \
        --mutations  mutations.tsv  \
        --parameters parameters.tsv \
        --posterior  posterior.tsv  \
        --elbo       elbo.tsv       \
        [--input     sg070_viber_input.tsv] \
        [--sample    sg070]

Expected files (written by extract_viber_rds.R + run_viber.R):
    mutations.tsv          : mutation_index  cluster  successes  trials  vaf
    parameters.tsv         : cluster  pi  theta
    posterior.tsv          : C1 C2 ... CK  mutation_index
    elbo.tsv               : iteration  ELBO
    <sample>_viber_input.tsv : mutation_id  successes  trials  (genomic coords)

Conda deps:
    conda install -c conda-forge matplotlib numpy pandas scipy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
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

def _cluster_order(par):
    return par.sort_values("theta", ascending=False)["cluster"].tolist()


def _palette(clusters):
    labels = sorted(clusters, key=lambda x: (int(x[1:]) if x[1:].isdigit() else 999))
    return {lbl: _BASE_COLOURS[i % len(_BASE_COLOURS)] for i, lbl in enumerate(labels)}


def _new_page(title, sample_id):
    fig, ax = plt.subplots(figsize=A4)
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=12, fontweight="bold", y=0.95)
    ax.set_position([PLOT_LEFT, PLOT_BOTTOM,
                     PLOT_RIGHT - PLOT_LEFT, PLOT_TOP - PLOT_BOTTOM])
    return fig, ax


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


def _beta_overlay(theta, n_obs, ax, colour):
    x = np.linspace(0, 1, 300)
    a = theta * 20 + 1
    b = (1 - theta) * 20 + 1
    y = beta_dist.pdf(x, a, b)
    scale = n_obs * 0.025 / max(y.max(), 1e-9)
    ax.plot(x, y * scale, color=colour, lw=2, alpha=0.85, zorder=5)

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


def load_data(mutations_path, parameters_path, posterior_path, elbo_path,
              input_path=None):
    mut  = _safe_read(mutations_path,  "mutations.tsv")
    par  = _safe_read(parameters_path, "parameters.tsv")
    post = _safe_read(posterior_path,  "posterior.tsv")
    elbo = _safe_read(elbo_path,       "elbo.tsv")

    if mut is None or par is None:
        sys.exit("ERROR: mutations.tsv and parameters.tsv are required.")

    for col in ("cluster", "successes", "trials"):
        if col not in mut.columns:
            sys.exit(f"ERROR: mutations.tsv missing column: {col}")

    mut["successes"] = pd.to_numeric(mut["successes"], errors="coerce")
    mut["trials"]    = pd.to_numeric(mut["trials"],    errors="coerce")
    bad = (mut["successes"].isna() | mut["trials"].isna() |
           (mut["trials"] <= 0)    | (mut["successes"] < 0))
    if bad.any():
        print(f"  [WARN] Dropping {bad.sum()} rows with invalid counts", file=sys.stderr)
        mut = mut[~bad].copy()

    mut["vaf"] = (pd.to_numeric(mut["vaf"], errors="coerce")
                  if "vaf" in mut.columns
                  else mut["successes"] / mut["trials"])

    if "mutation_index" not in mut.columns:
        mut["mutation_index"] = range(1, len(mut) + 1)
    mut["mutation_index"] = pd.to_numeric(mut["mutation_index"], errors="coerce")

    mut["mutation_id"] = pd.NA
    mut["chrom"]       = pd.NA
    mut["pos"]         = np.nan

    if input_path is not None:
        inp = _safe_read(input_path, "viber_input.tsv")
        if inp is not None and "mutation_id" in inp.columns:
            if len(inp) != len(mut):
                print(f"  [WARN] viber_input.tsv row count mismatch — skipping join",
                      file=sys.stderr)
            else:
                mut = mut.reset_index(drop=True)
                mut["mutation_id"] = inp["mutation_id"].values
                parts        = mut["mutation_id"].str.split(":", expand=True)
                mut["chrom"] = parts[0] if parts.shape[1] > 0 else pd.NA
                mut["pos"]   = (pd.to_numeric(parts[1], errors="coerce")
                                if parts.shape[1] > 1 else np.nan)
    else:
        print("  [INFO] --input not supplied; genomic plots will be skipped",
              file=sys.stderr)

    for col in ("cluster", "pi", "theta"):
        if col not in par.columns:
            sys.exit(f"ERROR: parameters.tsv missing column: {col}")
    par["pi"]    = pd.to_numeric(par["pi"],    errors="coerce")
    par["theta"] = pd.to_numeric(par["theta"], errors="coerce")
    par = par.dropna(subset=["pi", "theta"])

    if post is not None:
        if "mutation_index" not in post.columns:
            print("  [WARN] posterior.tsv has no mutation_index — skipping", file=sys.stderr)
            post = None
        else:
            post["mutation_index"] = pd.to_numeric(post["mutation_index"], errors="coerce")
            post = post.dropna(subset=["mutation_index"])

    if elbo is not None:
        elbo["ELBO"] = pd.to_numeric(elbo["ELBO"], errors="coerce")
        if "iteration" not in elbo.columns:
            elbo["iteration"] = range(1, len(elbo) + 1)

    return mut, par, post, elbo

# ── pages ──────────────────────────────────────────────────────────────────────

# ── p1: summary table ─────────────────────────────────────────────────────────
def page_summary_table(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    n_total = len(mut)
    k       = par["cluster"].nunique()
    cls_ord = [c for c in order if c in par_idx.index]

    fig, ax = _new_page("Summary — VIBER Clonal Clustering", sample_id)
    ax.axis("off")

    rows = [
        ["Total mutations",     str(n_total)],
        ["Clusters (k)",        str(k)],
        ["Method",              "Variational Bayesian mixture of Binomials"],
    ]
    for cl in cls_ord:
        theta  = par_idx.loc[cl, "theta"]
        pi_pct = par_idx.loc[cl, "pi"] * 100
        n_cl   = int((mut["cluster"] == cl).sum())
        ccf    = min(2 * theta, 1.0)
        cls    = ("Clonal" if theta >= 0.40
                  else "Subclonal" if theta >= 0.20
                  else "Low-CCF")
        rows.append([f"Cluster {cl}",
                     f"θ={theta:.3f}  π={pi_pct:.1f}%  CCF≈{ccf:.2f}  "
                     f"n={n_cl}  [{cls}]"])

    clonal    = [c for c in cls_ord if par_idx.loc[c,"theta"] >= 0.40]
    subclonal = [c for c in cls_ord if 0.20 <= par_idx.loc[c,"theta"] < 0.40]
    lowccf    = [c for c in cls_ord if par_idx.loc[c,"theta"] < 0.20]
    if clonal:
        rows.append(["► Clonal",    f"{', '.join(clonal)} — θ ≥ 0.40, founding clone"])
    if subclonal:
        rows.append(["► Subclonal", f"{', '.join(subclonal)} — 0.20 ≤ θ < 0.40"])
    if lowccf:
        rows.append(["► Low-CCF",   f"{', '.join(lowccf)} — θ < 0.20, rare / artefact?"])

    tbl = ax.table(cellText=rows, colLabels=["Property", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.2, 2.1)
    for j in range(2):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2): tbl[i, j].set_facecolor(shade)
    for i, cl in enumerate(cls_ord, start=4):
        tbl[i, 0].set_facecolor(pal.get(cl, "#CCCCCC"))
        tbl[i, 0].set_text_props(color="white", fontweight="bold")

    _caption(fig,
        "VIBER fits a Variational Bayesian mixture of Binomials to mutation read counts. "
        "Each cluster groups mutations sharing a similar VAF (θ). "
        "Under diploid CN-neutral loci: CCF ≈ 2θ. "
        "π = mixing proportion (fraction of all mutations in that cluster).")
    _save(pdf, fig)


# ── p2: θ lollipop ────────────────────────────────────────────────────────────
def page_theta_lollipop(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    cls_ord = [c for c in order if c in par_idx.index]

    fig, ax = _new_page("Fitted Binomial Peak θ per Cluster  (clonal first)", sample_id)

    y      = np.arange(len(cls_ord))
    thetas = [par_idx.loc[c, "theta"] for c in cls_ord]
    cols   = [pal.get(c, "#888888") for c in cls_ord]

    ax.hlines(y, 0, thetas, color=cols, lw=3, alpha=0.7)
    ax.scatter(thetas, y, color=cols, s=120, zorder=5)
    ax.axvline(0.50, color="#E74C3C", ls="--", lw=1.3, label="θ = 0.50  (clonal diploid)")
    ax.axvline(0.25, color="#F39C12", ls=":",  lw=1.1, label="θ = 0.25")
    ax.set_yticks(y); ax.set_yticklabels(cls_ord, fontsize=10)
    ax.set_xlim(0, 1.12); ax.legend(fontsize=9)
    for i, (c, t) in enumerate(zip(cls_ord, thetas)):
        ax.text(t + 0.02, i, f"{t:.3f}", va="center", fontsize=9, fontweight="bold")
    _style(ax, "Fitted θ per Cluster  (ordered by θ, clonal first)",
           "θ  (fitted VAF peak)", "Cluster")
    _caption(fig,
        "θ is VIBER's fitted Binomial peak — the modal VAF for mutations in each cluster. "
        "Under diploid CN-neutral loci, CCF ≈ 2θ. "
        "Clusters with θ ≈ 0.50 are likely clonal (present in the founding tumour cell).")
    _save(pdf, fig)


# ── p3: π bubble chart ────────────────────────────────────────────────────────
def page_pi_bubble(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    cls_ord = [c for c in order if c in par_idx.index]
    pis     = [par_idx.loc[c, "pi"] for c in cls_ord]

    fig, ax = _new_page("Mixing Proportions π per Cluster  (bubble area ∝ π)", sample_id)
    ax.set_xlim(-0.5, len(cls_ord) - 0.5)
    ax.set_ylim(-0.05, max(pis) * 1.45)

    for i, (c, pi) in enumerate(zip(cls_ord, pis)):
        ax.scatter(i, pi, s=pi * 5000, color=pal.get(c, "#888888"),
                   alpha=0.68, edgecolors="white", lw=1.5, zorder=3)
        ax.text(i, pi + max(pis) * 0.08,
                f"{c}\n{pi*100:.1f}%", ha="center", fontsize=9,
                fontweight="bold", color=pal.get(c, "#555555"))
    ax.set_xticks(range(len(cls_ord)))
    ax.set_xticklabels(cls_ord, fontsize=10)
    _style(ax, "Mixing Proportions π  (fraction of mutations per cluster)",
           "Cluster", "π  (mixing proportion)")
    _caption(fig,
        "π = mixing proportion: the fraction of all mutations attributed to each cluster "
        "by VIBER's variational posterior. Bubble area is proportional to π. "
        "This reflects relative cluster size, not CCF.")
    _save(pdf, fig)


# ── p4: CCF bar ───────────────────────────────────────────────────────────────
def page_ccf_bar(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    cls_ord = [c for c in order if c in par_idx.index]

    thetas = [par_idx.loc[c, "theta"] for c in cls_ord]
    ccfs   = [min(2 * t, 1.0) for t in thetas]
    ns     = [(mut["cluster"] == c).sum() for c in cls_ord]
    cols   = [pal.get(c, "#888888") for c in cls_ord]

    fig, ax = _new_page("Estimated Cancer Cell Fraction (CCF) per Cluster", sample_id)
    bars = ax.bar(cls_ord, ccfs, color=cols, edgecolor="white", alpha=0.88, width=0.55)
    ax.axhline(1.0, color="#E74C3C", ls="--", lw=1.3, label="CCF = 1.0  (clonal)")
    ax.axhline(0.5, color="#F39C12", ls=":",  lw=1.1, label="CCF = 0.5")
    for bar, ccf, n in zip(bars, ccfs, ns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                f"CCF≈{ccf:.2f}\n(n={n})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0, 1.38); ax.legend(fontsize=9)
    _style(ax, "CCF = min(2θ, 1.0) per Cluster  (diploid assumption)",
           "Cluster", "Cancer Cell Fraction (CCF)")
    _caption(fig,
        "CCF ≈ 2θ under diploid heterozygous CN-neutral loci (values capped at 1.0). "
        "Clonal clusters (CCF ≈ 1) were present in the founding tumour cell. "
        "Subclonal clusters arose later in a subset of cells.")
    _save(pdf, fig)


# ── p5: CCF schematic ─────────────────────────────────────────────────────────
def page_ccf_schematic(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    cls_ord = [c for c in order if c in par_idx.index]
    thetas  = [par_idx.loc[c, "theta"] for c in cls_ord]
    ccfs    = [min(2 * t, 1.0) for t in thetas]
    ns      = [(mut["cluster"] == c).sum() for c in cls_ord]

    fig, ax = _new_page("Clonal Architecture — Tumour Population Schematic", sample_id)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    cx, cy, max_r = 5.0, 5.0, 3.8
    sorted_arc = sorted(zip(cls_ord, ccfs, ns), key=lambda x: -x[1])
    n_circ = len(sorted_arc)
    for i, (cl, ccf, n) in enumerate(sorted_arc):
        r = max_r * ccf
        c = pal.get(cl, "#888888")
        ax.add_patch(plt.Circle((cx, cy), r, color=c, alpha=0.18, zorder=i))
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=c,
                                 lw=2.2, zorder=i+1))
        angle = (2 * np.pi * i / n_circ) - np.pi / 4
        lx = cx + r * 1.10 * np.cos(angle)
        ly = cy + r * 1.10 * np.sin(angle)
        ax.text(lx, ly, f"{cl}\nCCF≈{ccf:.2f}\nn={n}",
                ha="center", va="center", fontsize=9, color=c, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=c, alpha=0.88))
    ax.set_title("Tumour Cell Population  (circle area ∝ CCF)",
                 fontsize=12, fontweight="bold", pad=10)
    _caption(fig,
        "Schematic of nested tumour cell populations. Circle area ∝ CCF. "
        "Larger outer circles represent more prevalent clones. "
        "Nested structure implies subclones evolved within the founding clonal population.")
    _save(pdf, fig)


# ── p6+: VAF histogram — one page per cluster ─────────────────────────────────
def pages_vaf_per_cluster(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    bins    = np.linspace(0, 1, 41)

    for cl in order:
        sub    = mut[mut["cluster"] == cl]
        colour = pal.get(cl, "#888888")
        n_cl   = len(sub)
        theta  = par_idx.loc[cl, "theta"] if cl in par_idx.index else sub["vaf"].median()
        pi_pct = par_idx.loc[cl, "pi"] * 100 if cl in par_idx.index else np.nan

        fig, ax = _new_page(
            f"VAF Distribution — Cluster {cl}  "
            f"(θ={theta:.3f}  π={pi_pct:.1f}%  n={n_cl:,})",
            sample_id)

        ax.hist(sub["vaf"], bins=bins, color=colour, alpha=0.78, edgecolor="white")
        _beta_overlay(theta, n_cl, ax, colour)
        ax.axvline(theta, color="black", lw=2, ls="--", label=f"θ = {theta:.3f}")
        ax.axvline(0.5,   color="#BBBBBB", lw=1.2, ls=":", label="VAF = 0.5")
        ax.legend(fontsize=9)
        _style(ax, f"VAF Distribution — Cluster {cl}",
               "Variant Allele Frequency (VAF)", "Mutations")
        _caption(fig,
            f"VAF histogram for {n_cl} mutations in cluster {cl}. "
            "Black dashed = fitted Binomial peak θ (VIBER's variational posterior). "
            "Smooth curve = illustrative Beta density centred on θ. "
            "Grey dotted = VAF 0.5 (expected clonal heterozygous diploid VAF).")
        _save(pdf, fig)


# ── p: posterior confidence histogram ─────────────────────────────────────────
def page_posterior_hist(pdf, mut, par, post, sample_id, pal):
    if post is None:
        return
    order        = _cluster_order(par)
    cluster_cols = [c for c in order if c in post.columns]
    if not cluster_cols:
        return

    post_mat = post[cluster_cols].values
    max_post = post_mat.max(axis=1)
    conf_df  = pd.DataFrame({
        "mutation_index": post["mutation_index"].values,
        "max_post":       max_post,
    }).merge(mut[["mutation_index","vaf","cluster"]], on="mutation_index", how="left")

    fig, ax = _new_page("Posterior Assignment Confidence — Distribution", sample_id)
    for cl in order:
        sub = conf_df[conf_df["cluster"] == cl]["max_post"].dropna()
        if sub.empty: continue
        ax.hist(sub, bins=30, alpha=0.62, color=pal.get(cl,"#888888"),
                edgecolor="none", label=cl, range=(0, 1))
    ax.axvline(0.9, color="black", ls="--", lw=1.5, label="p = 0.90")
    ax.legend(fontsize=9)
    _style(ax, "Max Posterior Probability per Mutation  (all clusters)",
           "Max posterior probability  (r_nk)", "Mutation count")
    _caption(fig,
        "Distribution of assignment confidence scores (max r_nk) from VIBER's variational "
        "posterior. Values close to 1 indicate the mutation is unambiguously assigned. "
        "Values near 0.5 suggest the mutation sits between two clusters.")
    _save(pdf, fig)


# ── p: % high-confidence bar ──────────────────────────────────────────────────
def page_posterior_conf_bar(pdf, mut, par, post, sample_id, pal):
    if post is None:
        return
    order        = _cluster_order(par)
    cluster_cols = [c for c in order if c in post.columns]
    if not cluster_cols:
        return

    post_mat = post[cluster_cols].values
    max_post = post_mat.max(axis=1)
    conf_df  = pd.DataFrame({
        "mutation_index": post["mutation_index"].values,
        "max_post":       max_post,
    }).merge(mut[["mutation_index","cluster"]], on="mutation_index", how="left")

    cls_present = [c for c in order if c in conf_df["cluster"].unique()]
    pct_high    = [100 * (conf_df[conf_df["cluster"]==c]["max_post"] >= 0.9).mean()
                   for c in cls_present]

    fig, ax = _new_page("Assignment Confidence ≥ 0.90 per Cluster", sample_id)
    bars = ax.bar(cls_present, pct_high,
                  color=[pal.get(c,"#888888") for c in cls_present],
                  edgecolor="white", alpha=0.87)
    for bar, pct in zip(bars, pct_high):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{pct:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(90, color="#AAAAAA", ls=":", lw=1.2, label="90% threshold")
    ax.legend(fontsize=9)
    _style(ax, "% Mutations with Max Posterior ≥ 0.90",
           "Cluster", "% high-confidence mutations")
    _caption(fig,
        "Fraction of mutations assigned with ≥ 90% posterior confidence. "
        "Well-separated clusters score higher. Low fractions suggest overlap "
        "between clusters in VAF space.")
    _save(pdf, fig)


# ── p: VAF vs posterior scatter ───────────────────────────────────────────────
def page_vaf_vs_posterior(pdf, mut, par, post, sample_id, pal):
    if post is None:
        return
    order        = _cluster_order(par)
    cluster_cols = [c for c in order if c in post.columns]
    if not cluster_cols:
        return

    post_mat = post[cluster_cols].values
    max_post = post_mat.max(axis=1)
    conf_df  = pd.DataFrame({
        "mutation_index": post["mutation_index"].values,
        "max_post":       max_post,
    }).merge(mut[["mutation_index","vaf","cluster"]], on="mutation_index", how="left")

    fig, ax = _new_page("VAF vs Assignment Confidence per Mutation", sample_id)
    for cl in order:
        sub = conf_df[conf_df["cluster"]==cl].dropna(subset=["vaf","max_post"])
        ax.scatter(sub["vaf"], sub["max_post"],
                   color=pal.get(cl,"#888888"), s=12, alpha=0.45,
                   edgecolors="none", label=cl)
    ax.axhline(0.9, color="black", ls="--", lw=1.2, label="p = 0.90")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, markerscale=1.5, loc="lower right")
    _style(ax, "VAF vs Max Posterior Probability",
           "Variant Allele Frequency (VAF)", "Max posterior probability")
    _caption(fig,
        "Each dot = one mutation. Mutations near their cluster centre (θ) tend to have "
        "high posterior probability. Mutations between two cluster centres have "
        "lower confidence and appear below the dashed line.")
    _save(pdf, fig)


# ── p: posterior heatmap ──────────────────────────────────────────────────────
def page_posterior_heatmap(pdf, mut, par, post, sample_id, pal):
    if post is None:
        return
    order        = _cluster_order(par)
    cluster_cols = [c for c in order if c in post.columns]
    if not cluster_cols:
        return

    merged = (mut[["mutation_index","vaf","cluster"]]
              .merge(post, on="mutation_index", how="inner")
              .dropna(subset=cluster_cols)
              .sort_values(["cluster","vaf"]))

    MAX_ROWS = 500
    if len(merged) > MAX_ROWS:
        merged = (merged
                  .groupby("cluster", group_keys=False)
                  .apply(lambda g: g.sample(
                      min(len(g), max(1, int(MAX_ROWS*len(g)/len(merged)))),
                      random_state=42))
                  .sort_values(["cluster","vaf"]))

    # Custom layout: heatmap takes left 80%, colour strip right 10%
    fig = plt.figure(figsize=A4)
    fig.suptitle(f"Posterior Probability Heatmap (r_nk)  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.95)
    ax1 = fig.add_axes([PLOT_LEFT, PLOT_BOTTOM,
                        (PLOT_RIGHT-PLOT_LEFT)*0.82, PLOT_TOP-PLOT_BOTTOM])
    ax2 = fig.add_axes([(PLOT_LEFT + (PLOT_RIGHT-PLOT_LEFT)*0.84), PLOT_BOTTOM,
                        (PLOT_RIGHT-PLOT_LEFT)*0.11, PLOT_TOP-PLOT_BOTTOM])

    mat = merged[cluster_cols].values
    im  = ax1.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                     interpolation="nearest")
    ax1.set_xticks(range(len(cluster_cols)))
    ax1.set_xticklabels(cluster_cols, fontsize=9)
    ax1.set_yticks([])
    ax1.set_ylabel(f"Mutations  (n={len(merged)}, sorted by cluster then VAF)", fontsize=9)
    ax1.set_title("Posterior probability  r_nk", fontsize=11, fontweight="bold", pad=8)
    fig.colorbar(im, ax=ax1, shrink=0.5, label="Posterior probability")

    cl_int = np.array(pd.Categorical(merged["cluster"], categories=order).codes)
    cmap_s = mcolors.ListedColormap([pal.get(c,"#888888") for c in order])
    ax2.imshow(cl_int.reshape(-1,1), aspect="auto", cmap=cmap_s,
               vmin=0, vmax=len(order)-1, interpolation="nearest")
    ax2.set_yticks([]); ax2.set_xticks([0])
    ax2.set_xticklabels(["Cluster"], fontsize=8)
    ax2.set_title("Hard\nassignment", fontsize=9, fontweight="bold", pad=8)

    _caption(fig,
        "Rows = mutations (sorted by cluster then VAF); columns = VIBER clusters. "
        "Colour intensity = posterior probability r_nk. "
        "Clear block structure (one bright column per row) = well-separated clusters. "
        "Soft spread across columns = ambiguous assignments.")
    _save(pdf, fig)


# ── p: VAF vs depth (linear) ──────────────────────────────────────────────────
def page_vaf_depth_linear(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    fig, ax = _new_page("VAF vs Read Depth  (linear scale)", sample_id)

    for cl in order:
        sub = mut[mut["cluster"]==cl].dropna(subset=["vaf","trials"])
        ax.scatter(sub["trials"], sub["vaf"], color=pal.get(cl,"#888888"),
                   s=10, alpha=0.45, edgecolors="none", label=cl)
    for cl in order:
        if cl in par_idx.index:
            ax.axhline(par_idx.loc[cl,"theta"], color=pal.get(cl,"#888888"),
                       lw=1.0, ls="--", alpha=0.6)
    ax.axhline(0.5, color="#CCCCCC", lw=1.0, ls=":")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, markerscale=1.5, loc="upper right")
    _style(ax, "VAF vs Read Depth  (linear scale)",
           "Read depth (trials, ×)", "Variant Allele Frequency (VAF)")
    _caption(fig,
        "Each dot = one mutation, coloured by VIBER cluster. "
        "Dashed horizontal lines = fitted θ per cluster. "
        "Good clustering shows mutations scattered near their θ line with no depth bias.")
    _save(pdf, fig)


# ── p: VAF vs depth (log) ─────────────────────────────────────────────────────
def page_vaf_depth_log(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    fig, ax = _new_page("VAF vs Read Depth  (log scale)", sample_id)

    for cl in order:
        sub = mut[mut["cluster"]==cl].dropna(subset=["vaf","trials"])
        ax.scatter(sub["trials"], sub["vaf"], color=pal.get(cl,"#888888"),
                   s=10, alpha=0.45, edgecolors="none", label=cl)
    for cl in order:
        if cl in par_idx.index:
            ax.axhline(par_idx.loc[cl,"theta"], color=pal.get(cl,"#888888"),
                       lw=1.0, ls="--", alpha=0.6)
    ax.axhline(0.5, color="#CCCCCC", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, markerscale=1.5, loc="upper right")
    _style(ax, "VAF vs Read Depth  (log scale)",
           "Read depth (trials, ×)  [log]", "Variant Allele Frequency (VAF)")
    _caption(fig,
        "Log-scaled depth reveals low-coverage mutations (left side). "
        "Depth-dependent VAF drift may indicate mapping artefacts or CN effects.")
    _save(pdf, fig)


# ── p: within-cluster fit violin ──────────────────────────────────────────────
def page_fit_violin(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")

    mut2 = mut.copy()
    mut2["theta_cl"] = mut2["cluster"].map(
        {c: par_idx.loc[c,"theta"] for c in par_idx.index
         if c in mut2["cluster"].values})
    mut2["residual"] = (mut2["vaf"] - mut2["theta_cl"]).abs()
    mut2 = mut2.dropna(subset=["residual"])

    cls_present = [c for c in order if c in mut2["cluster"].values]
    data_by_cl  = [mut2[mut2["cluster"]==c]["residual"].values for c in cls_present]
    colours     = [pal.get(c,"#888888") for c in cls_present]

    fig, ax = _new_page("Within-Cluster Fit Quality  (|VAF − θ|)  — Violin", sample_id)

    valid = [(d,col) for d,col in zip(data_by_cl, colours) if len(d) > 1]
    if valid:
        positions = [i for i,d in enumerate(data_by_cl) if len(d) > 1]
        vp = ax.violinplot([d for d,_ in valid], positions=positions,
                           showmedians=True, showextrema=False)
        for body, (_,col) in zip(vp["bodies"], valid):
            body.set_facecolor(col); body.set_alpha(0.68)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2.2)
    ax.set_xticks(range(len(cls_present)))
    ax.set_xticklabels(cls_present, fontsize=10)
    ax.axhline(0.05, color="#AAAAAA", ls=":",  lw=1.0, label="|res| = 0.05")
    ax.axhline(0.10, color="#888888", ls="--", lw=1.0, label="|res| = 0.10")
    ax.legend(fontsize=9)
    _style(ax, "|VAF − θ| per Cluster  (deviation from cluster centre)",
           "Cluster", "|VAF − fitted θ|")
    _caption(fig,
        "Violin plot of |VAF − θ| per cluster. Narrow violins = tight, well-defined clusters. "
        "Broad or bimodal violins may indicate an unresolved mixture within a cluster.")
    _save(pdf, fig)


# ── p: within-cluster fit CDF ─────────────────────────────────────────────────
def page_fit_cdf(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")

    mut2 = mut.copy()
    mut2["theta_cl"] = mut2["cluster"].map(
        {c: par_idx.loc[c,"theta"] for c in par_idx.index
         if c in mut2["cluster"].values})
    mut2["residual"] = (mut2["vaf"] - mut2["theta_cl"]).abs()
    mut2 = mut2.dropna(subset=["residual"])

    cls_present = [c for c in order if c in mut2["cluster"].values]

    fig, ax = _new_page("Within-Cluster Fit Quality  (|VAF − θ|)  — CDF", sample_id)

    for cl in cls_present:
        resid = np.sort(mut2[mut2["cluster"]==cl]["residual"].values)
        if len(resid) == 0: continue
        cdf = np.arange(1, len(resid)+1) / len(resid)
        ax.plot(resid, cdf, color=pal.get(cl,"#888888"), lw=2.2, label=cl)
    ax.axvline(0.10, color="#AAAAAA", ls="--", lw=1.2, label="|res| = 0.10")
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9)
    _style(ax, "CDF of |VAF − θ| per Cluster",
           "|VAF − fitted θ|", "Cumulative fraction of mutations")
    _caption(fig,
        "Cumulative distribution of |VAF − θ| per cluster. "
        "CDFs rising steeply near 0 indicate tight clusters. "
        "The dotted line marks the 0.10 threshold — clusters where most mutations "
        "fall left of this are well-constrained.")
    _save(pdf, fig)


# ── p: ELBO trace ─────────────────────────────────────────────────────────────
def page_elbo_trace(pdf, elbo, sample_id):
    if elbo is None:
        return
    valid = elbo[np.isfinite(elbo["ELBO"])].copy()
    fig, ax = _new_page("ELBO Convergence Trace", sample_id)

    ax.plot(valid["iteration"], valid["ELBO"],
            color="#4C72B0", lw=2.2, marker="o", ms=3.5, zorder=3)
    ax.fill_between(valid["iteration"], valid["ELBO"],
                    valid["ELBO"].min(), alpha=0.12, color="#4C72B0")
    ax.text(0.97, 0.10, f"Final ELBO = {valid['ELBO'].iloc[-1]:.2f}",
            transform=ax.transAxes, ha="right", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#EBF5FB", edgecolor="#AED6F1"))
    _style(ax, "Evidence Lower Bound (ELBO) per Iteration",
           "Iteration", "ELBO")
    _caption(fig,
        "ELBO (Evidence Lower Bound) is the objective maximised by VIBER's variational EM. "
        "It should increase monotonically and plateau at convergence. "
        "A non-monotone trace may indicate numerical instability.")
    _save(pdf, fig)


# ── p: |ΔELBO| convergence rate ───────────────────────────────────────────────
def page_elbo_delta(pdf, elbo, sample_id):
    if elbo is None:
        return
    valid = elbo[np.isfinite(elbo["ELBO"])].copy()
    valid["delta"] = valid["ELBO"].diff().abs()
    d_valid = valid.dropna(subset=["delta"])

    fig, ax = _new_page("|ΔELBO| per Iteration  (convergence rate)", sample_id)
    ax.semilogy(d_valid["iteration"], d_valid["delta"],
                color="#DD8452", lw=2.2, marker="s", ms=3.5)
    ax.axhline(1e-10, color="#AAAAAA", ls="--", lw=1.2,
               label="ε = 1×10⁻¹⁰  (VIBER default)")
    ax.legend(fontsize=9)
    _style(ax, "|ΔELBO| per Iteration  (log scale)",
           "Iteration", "|ΔELBO|  (log scale)")
    _caption(fig,
        "|ΔELBO| per iteration on a log scale. "
        "Convergence is declared when |ΔELBO| drops below ε (default 1×10⁻¹⁰ in VIBER). "
        "Slow convergence may suggest increasing the number of EM restarts.")
    _save(pdf, fig)


# ── p: chromosome absolute ────────────────────────────────────────────────────
def page_chrom_absolute(pdf, mut, par, sample_id, pal):
    if mut["chrom"].isna().all():
        print("  [INFO] No genomic coordinates — skipping chromosome pages",
              file=sys.stderr)
        return
    order       = _cluster_order(par)
    present     = [c for c in CHR_ORDER if c in mut["chrom"].dropna().values]
    other       = sorted(set(mut["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other
    x           = np.arange(len(chrom_order))

    fig, ax = _new_page("Chromosomal Distribution — Absolute Counts", sample_id)
    ax.set_position([0.08, PLOT_BOTTOM, 0.87, PLOT_TOP - PLOT_BOTTOM])

    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (mut[mut["cluster"]==cl]
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
        "Absolute mutation counts per chromosome, stacked by VIBER cluster. "
        "Bar height = total mutations per chromosome.")
    _save(pdf, fig)


# ── p: chromosome 100% stacked ────────────────────────────────────────────────
def page_chrom_fraction(pdf, mut, par, sample_id, pal):
    if mut["chrom"].isna().all():
        return
    order       = _cluster_order(par)
    present     = [c for c in CHR_ORDER if c in mut["chrom"].dropna().values]
    other       = sorted(set(mut["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other
    x           = np.arange(len(chrom_order))
    totals      = mut.groupby("chrom").size().reindex(chrom_order, fill_value=0).values.astype(float)

    fig, ax = _new_page("Chromosomal Distribution — Cluster Composition (100% stacked)", sample_id)
    ax.set_position([0.08, PLOT_BOTTOM, 0.87, PLOT_TOP - PLOT_BOTTOM])

    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (mut[mut["cluster"]==cl]
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
        "Uniform composition across chromosomes supports genuine clonal structure. "
        "Strong enrichment of one cluster on a specific chromosome may indicate "
        "residual copy-number influence on the clustering.")
    _save(pdf, fig)


# ── p: genome-wide VAF ────────────────────────────────────────────────────────
def page_genome_vaf(pdf, mut, par, sample_id, pal):
    if mut["chrom"].isna().all() or mut["pos"].isna().all():
        return
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    present = [c for c in CHR_ORDER if c in mut["chrom"].dropna().values]

    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    mut2      = mut[mut["chrom"].isin(present)].copy()
    mut2["gx"] = mut2["chrom"].map(offsets).fillna(0) + mut2["pos"].fillna(0)

    fig, ax = _new_page("Genome-wide VAF Landscape", sample_id)
    for cl in order:
        sub = mut2[mut2["cluster"]==cl]
        ax.scatter(sub["gx"], sub["vaf"], color=pal.get(cl,"#888888"),
                   s=7, alpha=0.5, edgecolors="none", label=cl)
        if cl in par_idx.index:
            ax.axhline(par_idx.loc[cl,"theta"], color=pal.get(cl,"#888888"),
                       lw=0.9, ls="--", alpha=0.55)
    ax.axhline(0.5, color="#CCCCCC", lw=1.0, ls=":")
    for ch in present:
        ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        ax.text(mid, -0.07, ch.replace("chr",""),
                ha="center", fontsize=6, color="#555555",
                transform=ax.get_xaxis_transform())
    ax.set_xlim(0, cum); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right", ncol=2, markerscale=1.8)
    _style(ax, "VAF across the Genome  (coloured by VIBER cluster)",
           "Genomic position (hg38)", "Variant Allele Frequency (VAF)")
    _caption(fig,
        "Genome-wide VAF scatter coloured by cluster. "
        "Dashed lines = fitted θ per cluster. "
        "Regional VAF deviations from θ may indicate focal CN events not removed by the pre-filter.")
    _save(pdf, fig)


# ── p: genome-wide depth ──────────────────────────────────────────────────────
def page_genome_depth(pdf, mut, par, sample_id, pal):
    if mut["chrom"].isna().all() or mut["pos"].isna().all():
        return
    order   = _cluster_order(par)
    present = [c for c in CHR_ORDER if c in mut["chrom"].dropna().values]
    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    mut2       = mut[mut["chrom"].isin(present)].copy()
    mut2["gx"] = mut2["chrom"].map(offsets).fillna(0) + mut2["pos"].fillna(0)

    fig, ax = _new_page("Genome-wide Read Depth Landscape", sample_id)
    for cl in order:
        sub = mut2[mut2["cluster"]==cl]
        ax.scatter(sub["gx"], sub["trials"], color=pal.get(cl,"#888888"),
                   s=6, alpha=0.45, edgecolors="none", label=cl)
    med = mut2["trials"].median()
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
           "Genomic position (hg38)", "Read depth (trials, ×)")
    _caption(fig,
        "Read depth (= trials) per mutation coloured by cluster assignment. "
        "Depth should be roughly uniform; valleys may indicate mappability issues.")
    _save(pdf, fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIBER report — one plot per DIN A4 page."
    )
    parser.add_argument("--dir", "-d", default=None)
    parser.add_argument("--mutations",  default=None)
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--posterior",  default=None)
    parser.add_argument("--elbo",       default=None)
    parser.add_argument("--input",      default=None,
                        help="<sample>_viber_input.tsv (genomic coordinates)")
    parser.add_argument("--sample", "-s", default="sample")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    def _res(explicit, d, fname):
        if explicit: return explicit
        if d: return Path(d) / fname
        return None

    mutations_p  = _res(args.mutations,  args.dir, "mutations.tsv")
    parameters_p = _res(args.parameters, args.dir, "parameters.tsv")
    posterior_p  = _res(args.posterior,  args.dir, "posterior.tsv")
    elbo_p       = _res(args.elbo,       args.dir, "elbo.tsv")

    input_p = args.input
    if input_p is None and args.dir is not None:
        candidates = sorted(Path(args.dir).glob("*_viber_input.tsv"))
        if candidates:
            input_p = str(candidates[0])
            print(f"[viber_report] Auto-detected input TSV: {input_p}")
        else:
            print("[viber_report] No *_viber_input.tsv found — genomic plots skipped")

    if not mutations_p or not parameters_p:
        parser.error("Provide --dir  OR both --mutations and --parameters.")

    sid     = args.sample
    out_pdf = Path(args.output) if args.output else Path(f"{sid}_viber_report.pdf")

    print(f"[viber_report] Sample: {sid}")
    mut, par, post, elbo = load_data(
        mutations_p, parameters_p, posterior_p, elbo_p, input_p)

    print(f"[viber_report] {len(mut):,} mutations, "
          f"{par['cluster'].nunique()} clusters: {sorted(par['cluster'].tolist())}")
    print(f"[viber_report] posterior={'yes' if post is not None else 'missing'}, "
          f"elbo={'yes' if elbo is not None else 'missing'}, "
          f"genomic_coords={'yes' if not mut['chrom'].isna().all() else 'no'}")

    pal = _palette(par["cluster"].tolist())

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.5,
    })

    print(f"[viber_report] Writing: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        info = pdf.infodict()
        info["Title"]  = f"VIBER Interpretation Report — {sid}"
        info["Author"] = "viber_report.py"

        page_summary_table(pdf, mut, par, sid, pal)                   # summary table
        page_theta_lollipop(pdf, mut, par, sid, pal)                  # θ lollipop
        page_pi_bubble(pdf, mut, par, sid, pal)                       # π bubble
        page_ccf_bar(pdf, mut, par, sid, pal)                         # CCF bar
        page_ccf_schematic(pdf, mut, par, sid, pal)                   # CCF schematic
        pages_vaf_per_cluster(pdf, mut, par, sid, pal)                # VAF hist × cluster
        page_posterior_hist(pdf, mut, par, post, sid, pal)            # posterior hist
        page_posterior_conf_bar(pdf, mut, par, post, sid, pal)        # % conf bar
        page_vaf_vs_posterior(pdf, mut, par, post, sid, pal)          # VAF vs posterior
        page_posterior_heatmap(pdf, mut, par, post, sid, pal)         # r_nk heatmap
        page_vaf_depth_linear(pdf, mut, par, sid, pal)                # VAF vs depth linear
        page_vaf_depth_log(pdf, mut, par, sid, pal)                   # VAF vs depth log
        page_fit_violin(pdf, mut, par, sid, pal)                      # fit violin
        page_fit_cdf(pdf, mut, par, sid, pal)                         # fit CDF
        page_elbo_trace(pdf, elbo, sid)                               # ELBO trace
        page_elbo_delta(pdf, elbo, sid)                               # |ΔELBO|
        page_chrom_absolute(pdf, mut, par, sid, pal)                  # chrom absolute
        page_chrom_fraction(pdf, mut, par, sid, pal)                  # chrom 100%
        page_genome_vaf(pdf, mut, par, sid, pal)                      # genome VAF
        page_genome_depth(pdf, mut, par, sid, pal)                    # genome depth

    print(f"[viber_report] Done — {out_pdf}")


if __name__ == "__main__":
    main()