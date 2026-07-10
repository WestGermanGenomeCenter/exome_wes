#!/usr/bin/env python3
"""
viber_report.py — Visual interpretation report for VIBER clonal clustering.

Reads the four TSVs produced by extract_viber_rds.R and generates a multi-page
PDF that explains what VIBER found in biological terms.

Usage:
    python viber_report.py \
        --dir    /path/to/viber_export/  \
        [--sample sg070]                 \
        [--output sg070_viber_report.pdf]

    # Or supply files individually:
    python viber_report.py \
        --mutations  mutations.tsv  \
        --parameters parameters.tsv \
        --posterior  posterior.tsv  \
        --elbo       elbo.tsv       \
        [--sample sg070]

Expected TSV columns (written by extract_viber_rds.R):
    mutations.tsv   : mutation_id  mutation_index  cluster  successes  trials
    parameters.tsv  : cluster  pi  theta
    posterior.tsv   : C1 C2 ... CK  mutation_index
    elbo.tsv        : iteration  ELBO

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
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

# ── colour palette ─────────────────────────────────────────────────────────────
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

# ── helpers ────────────────────────────────────────────────────────────────────

def _cluster_order(par):
    """Sort clusters by fitted theta descending (most-clonal first)."""
    return par.sort_values("theta", ascending=False)["cluster"].tolist()


def _palette(clusters):
    labels = sorted(clusters, key=lambda x: (int(x[1:]) if x[1:].isdigit() else 999))
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


def _beta_overlay(theta, n_obs, ax, colour):
    """Beta density centred on theta, scaled to histogram peak."""
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


def load_data(mutations_path, parameters_path, posterior_path, elbo_path):
    mut  = _safe_read(mutations_path,  "mutations.tsv")
    par  = _safe_read(parameters_path, "parameters.tsv")
    post = _safe_read(posterior_path,  "posterior.tsv")
    elbo = _safe_read(elbo_path,       "elbo.tsv")

    if mut is None or par is None:
        sys.exit("ERROR: mutations.tsv and parameters.tsv are required.")

    # ── validate / clean mutations ─────────────────────────────────────────
    for col in ("mutation_id", "cluster", "successes", "trials"):
        if col not in mut.columns:
            sys.exit(f"ERROR: mutations.tsv missing column: {col}")

    mut["successes"] = pd.to_numeric(mut["successes"], errors="coerce")
    mut["trials"]    = pd.to_numeric(mut["trials"],    errors="coerce")
    bad = mut["successes"].isna() | mut["trials"].isna() | \
          (mut["trials"] <= 0)    | (mut["successes"] < 0)
    if bad.any():
        print(f"  [WARN] Dropping {bad.sum()} rows with invalid counts", file=sys.stderr)
        mut = mut[~bad].copy()

    mut["vaf"] = mut["successes"] / mut["trials"]

    parts = mut["mutation_id"].str.split(":", expand=True)
    mut["chrom"] = parts[0] if parts.shape[1] > 0 else pd.NA
    mut["pos"]   = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan

    if "mutation_index" not in mut.columns:
        mut["mutation_index"] = range(1, len(mut) + 1)
    mut["mutation_index"] = pd.to_numeric(mut["mutation_index"], errors="coerce")

    # ── validate parameters ────────────────────────────────────────────────
    for col in ("cluster", "pi", "theta"):
        if col not in par.columns:
            sys.exit(f"ERROR: parameters.tsv missing column: {col}")
    par["pi"]    = pd.to_numeric(par["pi"],    errors="coerce")
    par["theta"] = pd.to_numeric(par["theta"], errors="coerce")
    par = par.dropna(subset=["pi", "theta"])

    # ── validate posterior ─────────────────────────────────────────────────
    if post is not None:
        if "mutation_index" not in post.columns:
            print("  [WARN] posterior.tsv has no mutation_index — skipping", file=sys.stderr)
            post = None
        else:
            post["mutation_index"] = pd.to_numeric(post["mutation_index"], errors="coerce")
            post = post.dropna(subset=["mutation_index"])

    # ── validate elbo ──────────────────────────────────────────────────────
    if elbo is not None:
        elbo["ELBO"] = pd.to_numeric(elbo["ELBO"], errors="coerce")
        if "iteration" not in elbo.columns:
            elbo["iteration"] = range(1, len(elbo) + 1)

    return mut, par, post, elbo

# ── pages ──────────────────────────────────────────────────────────────────────

def page_summary(pdf, mut, par, elbo, sample_id, pal):
    order = _cluster_order(par)
    par_idx = par.set_index("cluster")
    n_total = len(mut)
    k = par["cluster"].nunique()

    fig = plt.figure(figsize=(13, 9))
    _header(fig, "VIBER — Summary of Clonal Clustering", sample_id)
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           top=0.91, bottom=0.10, left=0.05, right=0.97,
                           hspace=0.52, wspace=0.40)

    # ── top-left: parameter table ──────────────────────────────────────────
    ax_t = fig.add_subplot(gs[0, 0])
    ax_t.axis("off")
    rows = []
    cls_ord = [c for c in order if c in par_idx.index]
    for cl in cls_ord:
        pi_pct = par_idx.loc[cl, "pi"] * 100
        theta  = par_idx.loc[cl, "theta"]
        n_cl   = int((mut["cluster"] == cl).sum())
        ccf    = min(2 * theta, 1.0)
        interp = "Clonal" if theta >= 0.40 else ("Subclonal" if theta >= 0.20 else "Low-CCF")
        rows.append([cl, f"{pi_pct:.1f}%", f"{theta:.3f}", f"{ccf:.2f}", str(n_cl), interp])

    col_labels = ["Cluster", "π (fitted)", "θ (fitted)", "CCF est.", "# muts", "Class"]
    tbl = ax_t.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1.15, 1.85)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, cl in enumerate(cls_ord, start=1):
        tbl[i, 0].set_facecolor(pal.get(cl, "#CCCCCC"))
        tbl[i, 0].set_text_props(color="white", fontweight="bold")
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(1, len(col_labels)):
            tbl[i, j].set_facecolor(shade)
    ax_t.set_title("Fitted VIBER Parameters", fontsize=10, fontweight="bold", pad=6)

    # ── top-right: mixing-proportion pie ──────────────────────────────────
    ax_p = fig.add_subplot(gs[0, 1])
    pie_labels = [f"{c}\nθ={par_idx.loc[c,'theta']:.2f}" for c in cls_ord]
    pie_sizes  = [par_idx.loc[c, "pi"] for c in cls_ord]
    pie_cols   = [pal.get(c, "#888888") for c in cls_ord]
    wedges, texts, autos = ax_p.pie(
        pie_sizes, labels=pie_labels, colors=pie_cols,
        autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        textprops={"fontsize": 8},
    )
    for at in autos:
        at.set_fontsize(7.5); at.set_color("white"); at.set_fontweight("bold")
    ax_p.set_title("Mixing proportions (π)", fontsize=10, fontweight="bold", pad=6)

    # ── bottom-left: ELBO ─────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs[1, 0])
    if elbo is not None:
        valid = elbo[np.isfinite(elbo["ELBO"])]
        ax_e.plot(valid["iteration"], valid["ELBO"],
                  color="#4C72B0", lw=2, marker="o", ms=3)
        _style(ax_e, "ELBO Convergence", "Iteration", "ELBO", title_fs=10)
        ax_e.text(0.97, 0.08, f"Final ELBO = {valid['ELBO'].iloc[-1]:.1f}",
                  transform=ax_e.transAxes, ha="right", fontsize=8,
                  bbox=dict(boxstyle="round", facecolor="#EBF5FB", edgecolor="#AED6F1"))
    else:
        ax_e.text(0.5, 0.5, "elbo.tsv not available",
                  ha="center", va="center", transform=ax_e.transAxes, color="#888888")
        ax_e.axis("off")

    # ── bottom-right: plain-language blurb ───────────────────────────────
    ax_b = fig.add_subplot(gs[1, 1])
    ax_b.axis("off")
    clonal    = [c for c in cls_ord if par_idx.loc[c, "theta"] >= 0.40]
    subclonal = [c for c in cls_ord if 0.20 <= par_idx.loc[c, "theta"] < 0.40]
    lowccf    = [c for c in cls_ord if par_idx.loc[c, "theta"] < 0.20]

    lines = [
        f"VIBER fitted {k} cluster(s) to {n_total:,} somatic mutations using a",
        "Variational Bayesian mixture of Binomials. Each cluster groups mutations",
        "sharing a similar VAF, reflecting a shared Cancer Cell Fraction (CCF).\n",
        "Under diploid CN-neutral loci:  CCF ≈ 2 × θ\n",
    ]
    if clonal:
        lines.append(f"► CLONAL   {', '.join(clonal)}  (θ ≥ 0.40) — founding clone.")
    if subclonal:
        lines.append(f"► SUBCLONAL  {', '.join(subclonal)}  (0.20 ≤ θ < 0.40).")
    if lowccf:
        lines.append(f"► LOW-CCF   {', '.join(lowccf)}  (θ < 0.20) — rare / artefact?")

    ax_b.text(0.03, 0.97, "\n".join(lines), transform=ax_b.transAxes,
              ha="left", va="top", fontsize=9, color="#222222",
              bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB",
                        edgecolor="#AED6F1", alpha=0.85), wrap=True)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vaf_per_cluster(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    k = len(order)
    ncols = min(k, 3)
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows + 1.0),
                             squeeze=False)
    _header(fig, "VAF Distribution per Cluster  (with fitted θ)", sample_id)
    fig.subplots_adjust(hspace=0.65, wspace=0.38, top=0.91, bottom=0.13)

    bins = np.linspace(0, 1, 41)

    for idx, cl in enumerate(order):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub    = mut[mut["cluster"] == cl]
        colour = pal.get(cl, "#888888")
        n_cl   = len(sub)
        theta  = par_idx.loc[cl, "theta"] if cl in par_idx.index else sub["vaf"].median()
        pi_pct = par_idx.loc[cl, "pi"] * 100 if cl in par_idx.index else np.nan

        ax.hist(sub["vaf"], bins=bins, color=colour, alpha=0.75, edgecolor="white", zorder=2)
        _beta_overlay(theta, n_cl, ax, colour)
        ax.axvline(theta, color="black", lw=1.8, ls="--", label=f"θ = {theta:.3f}")
        ax.axvline(0.5,   color="#BBBBBB", lw=1.0, ls=":",  label="VAF 0.5")
        ax.legend(fontsize=7.5, loc="upper right")
        _style(ax, f"{cl}   π={pi_pct:.1f}%   n={n_cl:,}",
               "VAF", "Mutations", title_fs=9.5)

    for idx in range(k, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "Each panel: VAF histogram for one cluster. Black dashed = fitted Binomial peak θ "
        "(from VIBER's variational posterior). Smooth curve = illustrative Beta density "
        "centred on θ. Grey dotted = VAF 0.5 (clonal heterozygous diploid expectation). "
        "π = fitted mixing proportion; n = mutations hard-assigned to this cluster."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_theta_pi_overview(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    cls_ord = [c for c in order if c in par_idx.index]
    colours = [pal.get(c, "#888888") for c in cls_ord]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    _header(fig, "Fitted Cluster Parameters: θ and π", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.42)

    # lollipop: θ
    y = np.arange(len(cls_ord))
    thetas = [par_idx.loc[c, "theta"] for c in cls_ord]
    ax1.hlines(y, 0, thetas, color=colours, lw=2.5, alpha=0.7)
    ax1.scatter(thetas, y, color=colours, s=90, zorder=5)
    ax1.axvline(0.50, color="#E74C3C", ls="--", lw=1.2, label="θ = 0.50")
    ax1.axvline(0.25, color="#F39C12", ls=":",  lw=1.0, label="θ = 0.25")
    ax1.set_yticks(y); ax1.set_yticklabels(cls_ord, fontsize=9)
    ax1.set_xlim(0, 1.1)
    ax1.legend(fontsize=8)
    for i, (c, t) in enumerate(zip(cls_ord, thetas)):
        ax1.text(t + 0.02, i, f"{t:.3f}", va="center", fontsize=8.5, fontweight="bold")
    _style(ax1, "Fitted Binomial Peak θ per Cluster\n(ordered by θ, clonal first)",
           "θ  (fitted VAF peak)", "Cluster")

    # bubble: π
    pis = [par_idx.loc[c, "pi"] for c in cls_ord]
    ax2.set_xlim(-0.5, len(cls_ord) - 0.5); ax2.set_ylim(-0.05, max(pis) * 1.4)
    for i, (c, pi) in enumerate(zip(cls_ord, pis)):
        ax2.scatter(i, pi, s=pi * 4500, color=pal.get(c, "#888888"),
                    alpha=0.65, edgecolors="white", lw=1.5, zorder=3)
        ax2.text(i, pi + max(pis) * 0.07,
                 f"{c}\n{pi*100:.1f}%", ha="center", fontsize=8, fontweight="bold",
                 color=pal.get(c, "#555555"))
    ax2.set_xticks(range(len(cls_ord))); ax2.set_xticklabels(cls_ord, fontsize=9)
    _style(ax2, "Mixing Proportions π per Cluster\n(bubble area ∝ π)",
           "Cluster", "π  (mixing proportion)")

    _caption(fig,
        "Left: fitted Binomial peak θ — the modal VAF of each cluster according to the "
        "variational posterior. Under diploid CN-neutral loci, CCF ≈ 2θ. "
        "Right: mixing proportion π — the fraction of total mutations attributed to each cluster "
        "(bubble area proportional to π). "
        "θ and π are taken directly from parameters.tsv (fit$theta_k and fit$pi_k in R)."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_posterior_confidence(pdf, mut, par, post, sample_id, pal):
    if post is None:
        return

    order        = _cluster_order(par)
    cluster_cols = [c for c in order if c in post.columns]
    if not cluster_cols:
        return

    post_mat = post[cluster_cols].values
    max_post = post_mat.max(axis=1)
    assigned = np.array([cluster_cols[i] for i in post_mat.argmax(axis=1)])

    conf_df = pd.DataFrame({
        "mutation_index": post["mutation_index"].values,
        "max_post":       max_post,
        "post_cluster":   assigned,
    })
    conf_df = conf_df.merge(
        mut[["mutation_index", "vaf", "cluster"]],
        on="mutation_index", how="left"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    _header(fig, "Posterior Assignment Confidence", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.40)

    # histogram of max posterior per cluster
    ax = axes[0]
    for cl in order:
        sub = conf_df[conf_df["cluster"] == cl]["max_post"].dropna()
        if sub.empty: continue
        ax.hist(sub, bins=30, alpha=0.60, color=pal.get(cl, "#888888"),
                edgecolor="none", label=cl, range=(0, 1))
    ax.axvline(0.9, color="black", ls="--", lw=1.2, label="p = 0.90")
    ax.legend(fontsize=7.5)
    _style(ax, "Max Posterior per Mutation", "Max posterior probability", "Count")

    # % high-confidence per cluster
    ax2 = axes[1]
    cls_present = [c for c in order if c in conf_df["cluster"].unique()]
    pct_high = [100 * (conf_df[conf_df["cluster"] == c]["max_post"] >= 0.9).mean()
                for c in cls_present]
    bars = ax2.bar(cls_present, pct_high,
                   color=[pal.get(c, "#888888") for c in cls_present],
                   edgecolor="white", alpha=0.85)
    for bar, pct in zip(bars, pct_high):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5, f"{pct:.0f}%",
                 ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.axhline(90, color="#AAAAAA", ls=":", lw=1.0)
    _style(ax2, "% Mutations with\nmax posterior ≥ 0.90", "Cluster",
           "% high-confidence mutations")

    # scatter VAF vs max_post
    ax3 = axes[2]
    for cl in order:
        sub = conf_df[conf_df["cluster"] == cl].dropna(subset=["vaf", "max_post"])
        ax3.scatter(sub["vaf"], sub["max_post"], color=pal.get(cl, "#888888"),
                    s=10, alpha=0.45, edgecolors="none", label=cl)
    ax3.axhline(0.9, color="black", ls="--", lw=1.0)
    ax3.set_ylim(0, 1.05)
    ax3.legend(fontsize=7.5, markerscale=1.5, loc="lower right")
    _style(ax3, "VAF vs Assignment Confidence", "VAF", "Max posterior probability")

    _caption(fig,
        "r_nk from VIBER's variational posterior: probability of each mutation belonging to "
        "each cluster. The max across clusters is the assignment confidence. "
        "Left: distribution of confidence per cluster. "
        "Middle: fraction of mutations assigned with ≥ 90% confidence — well-separated clusters "
        "score higher. Right: confidence vs VAF — mutations near their cluster centre are "
        "typically more certain."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_posterior_heatmap(pdf, mut, par, post, sample_id, pal):
    if post is None:
        return

    order        = _cluster_order(par)
    cluster_cols = [c for c in order if c in post.columns]
    if not cluster_cols:
        return

    merged = mut[["mutation_index", "vaf", "cluster"]].merge(
        post, on="mutation_index", how="inner"
    ).dropna(subset=cluster_cols)
    merged = merged.sort_values(["cluster", "vaf"])

    MAX_ROWS = 500
    if len(merged) > MAX_ROWS:
        merged = (merged
                  .groupby("cluster", group_keys=False)
                  .apply(lambda g: g.sample(
                      min(len(g), max(1, int(MAX_ROWS * len(g) / len(merged)))),
                      random_state=42))
                  .sort_values(["cluster", "vaf"]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, max(6, len(merged) * 0.022 + 2)),
        gridspec_kw={"width_ratios": [4, 1]}
    )
    _header(fig, "Posterior Probability Heatmap (r_nk)", sample_id)
    fig.subplots_adjust(bottom=0.14, top=0.90, wspace=0.06)

    mat = merged[cluster_cols].values
    im  = ax1.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                     interpolation="nearest")
    ax1.set_xticks(range(len(cluster_cols)))
    ax1.set_xticklabels(cluster_cols, fontsize=8)
    ax1.set_yticks([])
    ax1.set_ylabel(f"Mutations  (n = {len(merged)}, sorted by cluster then VAF)", fontsize=9)
    ax1.set_title("Posterior probability  r_nk", fontsize=11, fontweight="bold", pad=8)
    plt.colorbar(im, ax=ax1, shrink=0.55, label="Posterior probability")

    # cluster colour strip
    cl_int  = np.array(pd.Categorical(merged["cluster"], categories=order).codes)
    cmap_s  = mcolors.ListedColormap([pal.get(c, "#888888") for c in order])
    ax2.imshow(cl_int.reshape(-1, 1), aspect="auto", cmap=cmap_s,
               vmin=0, vmax=len(order) - 1, interpolation="nearest")
    ax2.set_yticks([]); ax2.set_xticks([0])
    ax2.set_xticklabels(["Cluster"], fontsize=8)
    ax2.set_title("Hard\nassignment", fontsize=9, fontweight="bold", pad=8)

    _caption(fig,
        "Rows = mutations (sorted by cluster then VAF); columns = VIBER clusters. "
        "Colour intensity = posterior probability r_nk. "
        "Ideal clustering shows one bright column per row (clear block structure). "
        "Ambiguous mutations spread softly across multiple columns."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_ccf_architecture(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    cls_ord = [c for c in order if c in par_idx.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7))
    _header(fig, "Cancer Cell Fraction (CCF) & Clonal Architecture", sample_id)
    fig.subplots_adjust(bottom=0.18, top=0.89, wspace=0.42)

    thetas = [par_idx.loc[c, "theta"] for c in cls_ord]
    ccfs   = [min(2 * t, 1.0) for t in thetas]
    ns     = [(mut["cluster"] == c).sum() for c in cls_ord]
    cols   = [pal.get(c, "#888888") for c in cls_ord]

    bars = ax1.bar(cls_ord, ccfs, color=cols, edgecolor="white", alpha=0.88)
    ax1.axhline(1.0, color="#E74C3C", ls="--", lw=1.2, label="CCF = 1.0 (clonal)")
    ax1.axhline(0.5, color="#F39C12", ls=":",  lw=1.0, label="CCF = 0.5")
    for bar, ccf, n in zip(bars, ccfs, ns):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025,
                 f"CCF≈{ccf:.2f}\n(n={n})",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax1.set_ylim(0, 1.35)
    ax1.legend(fontsize=8)
    _style(ax1, "Estimated CCF per Cluster\n(CCF = min(2θ, 1.0), diploid assumed)",
           "Cluster", "Cancer Cell Fraction")

    # concentric-circle schematic
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10); ax2.axis("off")
    ax2.set_title("Tumour Cell Population Schematic  (area ∝ CCF)",
                  fontsize=10, fontweight="bold", pad=8)
    cx, cy, max_r = 5.0, 5.0, 3.8
    sorted_by_ccf = sorted(zip(cls_ord, ccfs, ns), key=lambda x: -x[1])
    n_circ = len(sorted_by_ccf)
    for i, (cl, ccf, n) in enumerate(sorted_by_ccf):
        r = max_r * ccf
        c = pal.get(cl, "#888888")
        ax2.add_patch(plt.Circle((cx, cy), r, color=c, alpha=0.18, zorder=i))
        ax2.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor=c, lw=2.0, zorder=i+1))
        angle = (2 * np.pi * i / n_circ) - np.pi / 4
        lx = cx + r * 1.08 * np.cos(angle)
        ly = cy + r * 1.08 * np.sin(angle)
        ax2.text(lx, ly, f"{cl}\nCCF≈{ccf:.2f}",
                 ha="center", va="center", fontsize=8, color=c, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                           edgecolor=c, alpha=0.85))

    _caption(fig,
        "CCF (Cancer Cell Fraction) = proportion of tumour cells carrying a mutation group. "
        "CCF ≈ 2θ under diploid heterozygous CN-neutral loci (values capped at 1.0). "
        "Clusters with CCF ≈ 1 were present in the founding cell (clonal). "
        "Lower-CCF clusters arose later in a subset of cells (subclonal). "
        "If VIBER input was filtered to CN-neutral segments (local_cn_a1 = local_cn_a2 = 1), "
        "this conversion is biologically valid."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_chromosome_distribution(pdf, mut, par, sample_id, pal):
    order       = _cluster_order(par)
    present     = [c for c in CHR_ORDER if c in mut["chrom"].dropna().values]
    other       = sorted(set(mut["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    _header(fig, "Chromosomal Distribution of Mutations by Cluster", sample_id)
    fig.subplots_adjust(hspace=0.58, bottom=0.12, top=0.91)

    x = np.arange(len(chrom_order))
    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (mut[mut["cluster"] == cl]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        ax1.bar(x, counts, bottom=bottom, label=cl,
                color=pal.get(cl, "#888888"), edgecolor="white", linewidth=0.3)
        bottom += counts
    ax1.set_xticks(x)
    ax1.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=8, loc="upper right")
    _style(ax1, "Mutations per Chromosome (stacked by cluster)", "Chromosome", "Count")

    totals = mut.groupby("chrom").size().reindex(chrom_order, fill_value=0).values.astype(float)
    bottom = np.zeros(len(chrom_order))
    for cl in order:
        counts = (mut[mut["cluster"] == cl]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        fracs  = np.where(totals > 0, counts / totals, 0)
        ax2.bar(x, fracs, bottom=bottom, label=cl,
                color=pal.get(cl, "#888888"), edgecolor="white", linewidth=0.3)
        bottom += fracs
    ax2.set_xticks(x)
    ax2.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc="upper right")
    _style(ax2, "Cluster Composition per Chromosome (100% stacked)", "Chromosome", "Fraction")

    _caption(fig,
        "Top: absolute mutation counts per chromosome, coloured by cluster. "
        "Bottom: proportion belonging to each cluster per chromosome. "
        "If clusters are biologically real, the composition (bottom) should be roughly "
        "uniform across chromosomes. Strong enrichment of one cluster on a specific chromosome "
        "may indicate residual copy-number influence.",
        y=0.03
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vaf_depth_scatter(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    _header(fig, "VAF vs. Read Depth", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.38)

    for ax, xscale, suffix in [
        (ax1, "linear", "linear scale"),
        (ax2, "log",    "log scale"),
    ]:
        for cl in order:
            sub = mut[mut["cluster"] == cl].dropna(subset=["vaf", "trials"])
            ax.scatter(sub["trials"], sub["vaf"], color=pal.get(cl, "#888888"),
                       s=10, alpha=0.45, edgecolors="none", label=cl)
        for cl in order:
            if cl in par_idx.index:
                ax.axhline(par_idx.loc[cl, "theta"], color=pal.get(cl, "#888888"),
                           lw=0.9, ls="--", alpha=0.55)
        ax.axhline(0.5, color="#CCCCCC", lw=1.0, ls=":")
        ax.set_xscale(xscale)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7.5, markerscale=1.5, loc="upper right")
        _style(ax, f"VAF vs Depth — {suffix}",
               "Read depth (trials, ×)" + ("  [log]" if xscale == "log" else ""), "VAF")

    _caption(fig,
        "Each dot = one mutation, coloured by cluster. Dashed horizontal lines = fitted θ per cluster. "
        "Good clustering shows mutations scattered near their θ line, independent of depth. "
        "Depth-dependent VAF bias may indicate mapping or CN artefacts. "
        "Right panel uses log depth to reveal low-coverage mutations."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_within_cluster_fit(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")

    mut2 = mut.copy()
    mut2["theta_cl"] = mut2["cluster"].map(
        {c: par_idx.loc[c, "theta"] for c in par_idx.index if c in mut2["cluster"].values}
    )
    mut2["residual"] = (mut2["vaf"] - mut2["theta_cl"]).abs()
    mut2 = mut2.dropna(subset=["residual"])

    cls_present = [c for c in order if c in mut2["cluster"].values]
    data_by_cl  = [mut2[mut2["cluster"] == c]["residual"].values for c in cls_present]
    colours     = [pal.get(c, "#888888") for c in cls_present]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    _header(fig, "Within-Cluster Fit Quality  (|VAF − θ|)", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.40)

    valid_for_violin = [(d, c) for d, c in zip(data_by_cl, colours) if len(d) > 1]
    if valid_for_violin:
        positions = [i for i, d in enumerate(data_by_cl) if len(d) > 1]
        vp = ax1.violinplot([d for d, _ in valid_for_violin],
                            positions=positions,
                            showmedians=True, showextrema=False)
        for body, (_, col) in zip(vp["bodies"], valid_for_violin):
            body.set_facecolor(col); body.set_alpha(0.65)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
    ax1.set_xticks(range(len(cls_present)))
    ax1.set_xticklabels(cls_present, fontsize=8)
    ax1.axhline(0.05, color="#AAAAAA", ls=":",  lw=1.0, label="|res| = 0.05")
    ax1.axhline(0.10, color="#888888", ls="--", lw=1.0, label="|res| = 0.10")
    ax1.legend(fontsize=8)
    _style(ax1, "Residual |VAF − θ| per Cluster", "Cluster", "|VAF − fitted θ|")

    for cl, col in zip(cls_present, colours):
        resid = np.sort(mut2[mut2["cluster"] == cl]["residual"].values)
        if len(resid) == 0: continue
        cdf = np.arange(1, len(resid) + 1) / len(resid)
        ax2.plot(resid, cdf, color=col, lw=2, label=cl)
    ax2.axvline(0.10, color="#AAAAAA", ls="--", lw=1.0)
    ax2.set_xlim(0, 0.5); ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.legend(fontsize=8)
    _style(ax2, "CDF of |VAF − θ| per Cluster",
           "|VAF − fitted θ|", "Cumulative fraction")

    _caption(fig,
        "How tightly mutations cluster around their fitted θ. "
        "Narrow violins and CDFs rising steeply near 0 = tight, well-defined clusters. "
        "Broad distributions may indicate a cluster spans two subclones, "
        "or residual CN-driven VAF variance."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_elbo_detail(pdf, elbo, sample_id):
    if elbo is None:
        return

    valid = elbo[np.isfinite(elbo["ELBO"])].copy()
    valid["delta"] = valid["ELBO"].diff().abs()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    _header(fig, "ELBO Convergence — Detail", sample_id)
    fig.subplots_adjust(bottom=0.22, top=0.89, wspace=0.40)

    ax1.plot(valid["iteration"], valid["ELBO"],
             color="#4C72B0", lw=2, marker="o", ms=3, zorder=3)
    ax1.fill_between(valid["iteration"], valid["ELBO"],
                     valid["ELBO"].min(), alpha=0.12, color="#4C72B0")
    _style(ax1, "Evidence Lower Bound (ELBO) per Iteration",
           "Iteration", "ELBO")
    ax1.text(0.97, 0.10, f"Final ELBO = {valid['ELBO'].iloc[-1]:.2f}",
             transform=ax1.transAxes, ha="right", fontsize=9, fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#EBF5FB", edgecolor="#AED6F1"))

    d_valid = valid.dropna(subset=["delta"])
    ax2.semilogy(d_valid["iteration"], d_valid["delta"],
                 color="#DD8452", lw=2, marker="s", ms=3)
    ax2.axhline(1e-10, color="#AAAAAA", ls="--", lw=1.0, label="ε = 1e-10")
    ax2.legend(fontsize=8)
    _style(ax2, "|ΔELBO| per Iteration  (convergence rate)",
           "Iteration", "|ΔELBO|  (log scale)")

    _caption(fig,
        "ELBO = Evidence Lower Bound — the objective maximised by VIBER's variational EM. "
        "Left: should increase monotonically and plateau. "
        "Right: |ΔELBO| per step on a log scale; convergence is declared when this "
        "drops below ε (default 1×10⁻¹⁰ in VIBER). "
        "A non-monotone or oscillating ELBO trace may indicate that more EM starts "
        "(samples parameter) are needed."
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_genome_landscape(pdf, mut, par, sample_id, pal):
    order   = _cluster_order(par)
    par_idx = par.set_index("cluster")
    present = [c for c in CHR_ORDER if c in mut["chrom"].dropna().values]

    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum
        cum += CHR_LENGTHS.get(ch, 100e6)

    mut2 = mut[mut["chrom"].isin(present)].copy()
    mut2["gx"] = (mut2["chrom"].map(offsets).fillna(0) +
                  mut2["pos"].fillna(0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    _header(fig, "Genome-wide Mutation Landscape", sample_id)
    fig.subplots_adjust(hspace=0.10, bottom=0.13, top=0.91)

    for cl in order:
        sub = mut2[mut2["cluster"] == cl]
        ax1.scatter(sub["gx"], sub["vaf"], color=pal.get(cl, "#888888"),
                    s=8, alpha=0.5, edgecolors="none", label=cl)
    for cl in order:
        if cl in par_idx.index:
            ax1.axhline(par_idx.loc[cl, "theta"],
                        color=pal.get(cl, "#888888"), lw=0.8, ls="--", alpha=0.5)
    ax1.axhline(0.5, color="#CCCCCC", lw=1.0, ls=":")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=7.5, markerscale=1.5, loc="upper right", ncol=2)
    _style(ax1, "VAF across the Genome (coloured by cluster)", "", "VAF")

    ax2.scatter(mut2["gx"], mut2["trials"],
                color="#888888", s=5, alpha=0.3, edgecolors="none")
    _style(ax2, "Read Depth across the Genome",
           "Genomic position (hg38)", "Depth (×)")

    for ch in present:
        for ax in (ax1, ax2):
            ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        ax2.text(mid, ax2.get_ylim()[1] * 0.90,
                 ch.replace("chr", ""), ha="center", fontsize=6, color="#555555")

    ax1.set_xlim(0, cum)
    _caption(fig,
        "Genome-wide view of somatic mutations. "
        "Top: VAF coloured by cluster; dashed lines = fitted θ per cluster. "
        "Bottom: read depth per site. "
        "VAF deviations from θ in contiguous genomic regions may indicate "
        "focal copy-number events not removed by the CN pre-filter.",
        y=0.03
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="VIBER interpretation report from extracted TSVs."
    )
    parser.add_argument("--dir", "-d", default=None,
                        help="Directory with mutations.tsv / parameters.tsv / "
                             "posterior.tsv / elbo.tsv")
    parser.add_argument("--mutations",  default=None, help="mutations.tsv path")
    parser.add_argument("--parameters", default=None, help="parameters.tsv path")
    parser.add_argument("--posterior",  default=None, help="posterior.tsv path")
    parser.add_argument("--elbo",       default=None, help="elbo.tsv path")
    parser.add_argument("--sample", "-s", default="sample",
                        help="Sample ID for plot titles")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PDF (default: <sample>_viber_report.pdf)")
    args = parser.parse_args()

    def _res(explicit, d, fname):
        if explicit: return explicit
        if d: return Path(d) / fname
        return None

    mutations_p  = _res(args.mutations,  args.dir, "mutations.tsv")
    parameters_p = _res(args.parameters, args.dir, "parameters.tsv")
    posterior_p  = _res(args.posterior,  args.dir, "posterior.tsv")
    elbo_p       = _res(args.elbo,       args.dir, "elbo.tsv")

    if not mutations_p or not parameters_p:
        parser.error("Provide --dir  OR both --mutations and --parameters.")

    sample_id = args.sample
    out_pdf   = Path(args.output) if args.output \
                else Path(f"{sample_id}_viber_report.pdf")

    print(f"[viber_report] Sample: {sample_id}")
    mut, par, post, elbo = load_data(mutations_p, parameters_p, posterior_p, elbo_p)
    print(f"[viber_report] {len(mut):,} mutations, "
          f"{par['cluster'].nunique()} clusters: {sorted(par['cluster'].tolist())}")
    print(f"[viber_report] posterior={'yes' if post is not None else 'missing'}, "
          f"elbo={'yes' if elbo is not None else 'missing'}")

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
        d = pdf.infodict()
        d["Title"]  = f"VIBER Interpretation Report — {sample_id}"
        d["Author"] = "viber_report.py"

        page_summary(pdf, mut, par, elbo, sample_id, pal)               # p1
        page_vaf_per_cluster(pdf, mut, par, sample_id, pal)             # p2
        page_theta_pi_overview(pdf, mut, par, sample_id, pal)           # p3
        page_posterior_confidence(pdf, mut, par, post, sample_id, pal)  # p4
        page_posterior_heatmap(pdf, mut, par, post, sample_id, pal)     # p5
        page_ccf_architecture(pdf, mut, par, sample_id, pal)            # p6
        page_chromosome_distribution(pdf, mut, par, sample_id, pal)     # p7
        page_vaf_depth_scatter(pdf, mut, par, sample_id, pal)           # p8
        page_within_cluster_fit(pdf, mut, par, sample_id, pal)          # p9
        page_elbo_detail(pdf, elbo, sample_id)                          # p10
        page_genome_landscape(pdf, mut, par, sample_id, pal)            # p11

    print(f"[viber_report] Done — {out_pdf}")


if __name__ == "__main__":
    main()