#!/usr/bin/env python3
"""
orchard_report.py — Visual interpretation report for Orchard phylogenetic output.

One plot per page, all pages DIN A4 (8.27 × 11.69 inches).

Usage:
    python orchard_report.py \\
        --npz    st015.orchard.npz \\
        --params st015.params.json \\
        --ssm    st015.ssm \\
        [--sample st015] \\
        [--output st015_orchard_report.pdf]

Input files:
    *.orchard.npz   — Orchard archive (struct, phi, eta, llh, prob, newick, ...)
    *.params.json   — cluster→mutation mapping + sample names
    *.ssm           — mutation-level read counts (id, name, var_reads, total_reads)

NPZ key reference (Orchard / Pairtree format):
    struct   (n_trees, K)          parent vector; struct[t,k] = parent of node k+1
    phi      (n_trees, K+1, S)     population frequency (fraction of ALL cells)
    eta      (n_trees, K+1, S)     clone-exclusive frequency
    llh      (n_trees,)            log-likelihood, sorted best→worst
    prob     (n_trees,)            posterior probability per tree
    count    (n_trees,)            times sampled
    newick   (n_trees,)            Newick string (node labels = 0..K; 0 = normal root)
    clusters.json                  list of K lists of mutation IDs
    sampnames.json                 sample name(s)

Conda deps:
    conda install -c conda-forge matplotlib numpy pandas scipy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

# ── DIN A4 dimensions (inches) ─────────────────────────────────────────────────
A4 = (8.27, 11.69)

# Layout constants: leave room for suptitle at top and caption at bottom
PLOT_TOP    = 0.88   # axes top edge  (suptitle lives 0.91–1.0)
PLOT_BOTTOM = 0.14   # axes bottom edge (caption lives 0.02–0.11)
PLOT_LEFT   = 0.10
PLOT_RIGHT  = 0.95

# ── colour palette ─────────────────────────────────────────────────────────────
_NODE_COLOURS = [
    "#AAAAAA",   # node 0 = normal/germline root
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#64B5CD", "#CCB974", "#E377C2",
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

def _nc(n):
    return _NODE_COLOURS[n % len(_NODE_COLOURS)]


def _new_page(title, sample_id):
    """Create a fresh A4 figure with suptitle. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=A4)
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=12, fontweight="bold", y=0.95)
    ax.set_position([PLOT_LEFT, PLOT_BOTTOM, PLOT_RIGHT - PLOT_LEFT,
                     PLOT_TOP - PLOT_BOTTOM])
    return fig, ax


def _new_page_noax(title, sample_id):
    """A4 figure with suptitle but no default axes (caller adds subplots manually)."""
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
    """Italic caption in the reserved bottom strip."""
    fig.text(PLOT_LEFT, 0.02, text,
             ha="left", va="bottom", fontsize=fontsize,
             fontstyle="italic", color="#444444", wrap=True,
             transform=fig.transFigure)


def _save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


# ── JSON helper ────────────────────────────────────────────────────────────────

def _load_json_key(npz, key):
    raw = npz[key]
    if isinstance(raw, bytes):
        try:
            return json.loads(raw.decode())
        except Exception:
            return None
    try:
        return json.loads(bytes(raw).decode())
    except Exception:
        return None

# ── data loading ───────────────────────────────────────────────────────────────

def load_data(npz_path, params_path, ssm_path):
    try:
        npz = np.load(str(npz_path), allow_pickle=True)
    except Exception as e:
        sys.exit(f"ERROR loading NPZ: {e}")

    for k in {"struct", "phi", "llh", "prob", "newick"}:
        if k not in npz.keys():
            sys.exit(f"ERROR: NPZ missing key: {k}")

    struct = npz["struct"]
    phi    = npz["phi"]
    eta    = npz["eta"] if "eta" in npz else None
    llh    = npz["llh"]
    prob   = npz["prob"]
    count  = npz["count"] if "count" in npz else np.ones(len(llh), dtype=int)
    newick = npz["newick"]

    n_trees, K1, S = phi.shape
    K = K1 - 1

    clusters_npz  = _load_json_key(npz, "clusters.json")  if "clusters.json"  in npz.keys() else None
    sampnames_npz = _load_json_key(npz, "sampnames.json") if "sampnames.json" in npz.keys() else None

    params = {}
    if params_path and Path(params_path).exists():
        try:
            with open(params_path) as f:
                params = json.load(f)
        except Exception as e:
            print(f"  [WARN] params.json: {e}", file=sys.stderr)

    clusters = params.get("clusters") or clusters_npz or []
    samples  = params.get("samples")  or sampnames_npz or "unknown"
    if isinstance(samples, str):
        samples = [samples]

    ssm = None
    if ssm_path and Path(ssm_path).exists():
        try:
            ssm = pd.read_csv(ssm_path, sep="\t")
            for c in ("var_reads", "total_reads"):
                if c in ssm.columns:
                    ssm[c] = pd.to_numeric(ssm[c], errors="coerce")
            ssm["vaf"]   = ssm["var_reads"] / ssm["total_reads"]
            parts        = ssm["name"].str.split(":", expand=True)
            ssm["chrom"] = parts[0] if parts.shape[1] > 0 else pd.NA
            ssm["pos"]   = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
        except Exception as e:
            print(f"  [WARN] SSM: {e}", file=sys.stderr)
    else:
        print("  [INFO] --ssm not supplied; mutation-level plots skipped", file=sys.stderr)

    # attach cluster index to SSM rows
    if ssm is not None and clusters:
        id2cl = {}
        for k, cl in enumerate(clusters):
            for sid in cl:
                id2cl[sid] = k
        ssm["cluster"] = ssm["id"].map(id2cl)

    return dict(struct=struct, phi=phi, eta=eta, llh=llh, prob=prob,
                count=count, newick=newick, n_trees=n_trees, K=K, S=S,
                samples=samples, clusters=clusters, ssm=ssm)

# ── tree drawing primitive ─────────────────────────────────────────────────────

def _build_children(sv):
    K = len(sv)
    ch = {i: [] for i in range(K + 1)}
    for k, p in enumerate(sv):
        node, parent = k + 1, int(p)
        if parent == node:
            parent = 0
        ch[parent].append(node)
    return ch


def _layout(children, root=0):
    pos, ctr = {}, [0]
    def place(n, d):
        kids = children.get(n, [])
        if not kids:
            pos[n] = (ctr[0], -d); ctr[0] += 1
        else:
            for c in kids:
                place(c, d + 1)
            pos[n] = (np.mean([pos[c][0] for c in kids]), -d)
    place(root, 0)
    return pos


def _draw_tree(ax, struct_vec, phi_vec, eta_vec, clusters,
               title="", node_fs=7, phi_fs=7.5, annotate_eta=True):
    """Draw one Orchard tree on ax. Returns node_pos dict."""
    K        = len(struct_vec)
    children = _build_children(struct_vec)
    raw_pos  = _layout(children)

    xs = np.array([raw_pos[n][0] for n in range(K + 1)], dtype=float)
    ys = np.array([raw_pos[n][1] for n in range(K + 1)], dtype=float)

    if xs.max() > xs.min():
        xs = (xs - xs.min()) / (xs.max() - xs.min())
    xs = xs * 0.78 + 0.11

    ys -= ys.min()
    if ys.max() > 0:
        ys /= ys.max()
    ys = ys * 0.72 + 0.14

    npos = {n: (xs[n], ys[n]) for n in range(K + 1)}

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    # edges
    for node in range(K + 1):
        for child in children.get(node, []):
            x0, y0 = npos[node]
            x1, y1 = npos[child]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#555555",
                                        lw=1.8, mutation_scale=14))

    # nodes
    for node in range(K + 1):
        x, y   = npos[node]
        col    = _nc(node)
        phi_v  = float(phi_vec[node]) if node < len(phi_vec) else 0.0
        radius = 0.05 + phi_v * 0.09

        ax.add_patch(Circle((x, y), radius, color=col, alpha=0.85, zorder=4,
                             transform=ax.transData))
        ax.add_patch(Circle((x, y), radius, fill=False, edgecolor="white",
                             lw=1.2, zorder=5, transform=ax.transData))

        if node == 0:
            label = "Normal\nroot"
        else:
            n_m = len(clusters[node - 1]) if (node - 1) < len(clusters) else "?"
            label = f"C{node}\n({n_m} muts)"

        ax.text(x, y, label, ha="center", va="center",
                fontsize=node_fs, color="white", fontweight="bold", zorder=6)

        # φ below node
        if node > 0:
            ax.text(x, y - radius - 0.04, f"φ={phi_v:.3f}",
                    ha="center", va="top", fontsize=phi_fs, color=col)

        # η above node
        if annotate_eta and eta_vec is not None and node < len(eta_vec):
            eta_v = float(eta_vec[node])
            ax.text(x, y + radius + 0.02, f"η={eta_v:.3f}",
                    ha="center", va="bottom", fontsize=phi_fs,
                    color=col, fontstyle="italic")

    return npos

# ── individual pages ───────────────────────────────────────────────────────────

# ── p1: run summary table ──────────────────────────────────────────────────────
def page_summary_table(pdf, d, sample_id):
    fig, ax = _new_page("Summary — Run Statistics", sample_id)
    ax.axis("off")

    clusters = d["clusters"]
    K        = d["K"]
    phi_best = d["phi"][0]
    n_total  = sum(len(c) for c in clusters)

    rows = [
        ["Sample(s)",           ", ".join(d["samples"])],
        ["Tumour clusters (K)", str(K)],
        ["Total mutations",     str(n_total)],
        ["Trees sampled",       str(d["n_trees"])],
        ["Best tree LLH",       f"{d['llh'][0]:.4f}"],
        ["Best tree posterior", f"{d['prob'][0]:.4f}"],
        ["Best tree count",     str(int(d["count"][0]))],
        ["Best tree Newick",    str(d["newick"][0])],
    ]
    for i, cl in enumerate(clusters):
        phi_v = float(phi_best[i + 1, 0])
        ccf   = min(phi_v * 2, 1.0)
        cls   = "Clonal" if phi_v >= 0.40 else ("Subclonal" if phi_v >= 0.15 else "Low-CCF")
        rows.append([f"Cluster C{i+1}",
                     f"φ={phi_v:.4f}  CCF≈{ccf:.3f}  n={len(cl)}  [{cls}]"])

    tbl = ax.table(cellText=rows, colLabels=["Property", "Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.2)
    for j in range(2):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2):
            tbl[i, j].set_facecolor(shade)
    for i in range(len(rows) - K, len(rows)):
        node = i - (len(rows) - K) + 1
        tbl[i + 1, 0].set_facecolor(_nc(node))
        tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")

    _caption(fig,
        "Orchard infers a ranked ensemble of tumour phylogenetic trees using a branch-and-bound "
        "algorithm. Trees are sorted by log-likelihood (best first). "
        "φ = population frequency (fraction of ALL cells carrying that cluster's mutations). "
        "CCF ≈ 2φ under diploid heterozygous copy-neutral loci.")
    _save(pdf, fig)


# ── p2: best tree (large, annotated) ──────────────────────────────────────────
def page_best_tree(pdf, d, sample_id):
    fig, ax = _new_page(
        f"Best Tree  (LLH={d['llh'][0]:.4f}  prob={d['prob'][0]:.4f}  "
        f"Newick: {d['newick'][0]})", sample_id)

    eta0 = d["eta"][0, :, 0] if d["eta"] is not None else None
    _draw_tree(ax, d["struct"][0], d["phi"][0, :, 0], eta0,
               d["clusters"], node_fs=9, phi_fs=8.5, annotate_eta=True)

    _caption(fig,
        "Best tree by log-likelihood. Bubble area ∝ φ (population frequency). "
        "φ values are shown below each node; η (clone-exclusive fraction) above. "
        "Arrows denote parent→child evolutionary relationships. "
        "Node 0 (grey) = normal/germline root, always φ=1.")
    _save(pdf, fig)


# ── p3–pN: one page per additional tree ───────────────────────────────────────
def pages_all_trees(pdf, d, sample_id):
    for i in range(d["n_trees"]):
        llh_v  = d["llh"][i]
        prob_v = d["prob"][i]
        cnt_v  = int(d["count"][i])
        nwk    = d["newick"][i]
        fig, ax = _new_page(
            f"Tree T{i}  |  LLH={llh_v:.4f}  prob={prob_v:.4e}  "
            f"count={cnt_v}  Newick: {nwk}", sample_id)
        eta_i = d["eta"][i, :, 0] if d["eta"] is not None else None
        _draw_tree(ax, d["struct"][i], d["phi"][i, :, 0], eta_i,
                   d["clusters"], node_fs=9, phi_fs=8.5, annotate_eta=True)
        _caption(fig,
            f"Tree T{i} from the Orchard ensemble (T0 = best). "
            "Bubble area ∝ φ. Trees are sorted by LLH; lower = better.")
        _save(pdf, fig)


# ── p: LLH bar chart ──────────────────────────────────────────────────────────
def page_llh(pdf, d, sample_id):
    fig, ax = _new_page("Tree Log-Likelihoods (ΔLLH relative to best)", sample_id)
    n = d["n_trees"]
    x = np.arange(n)
    delta = d["llh"] - d["llh"][0]
    bars  = ax.bar(x, delta, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.axhline(0, color="#E74C3C", ls="--", lw=1.2, label="Best tree (T0)")
    for bar, dv in zip(bars, delta):
        if dv < -0.5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    dv - abs(dv) * 0.02, f"{dv:.2f}",
                    ha="center", va="top", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n)], fontsize=9)
    ax.legend(fontsize=9)
    _style(ax, "ΔLLH per Tree  (LLHᵢ − LLH_best)",
           "Tree index", "ΔLLH")
    _caption(fig,
        "Log-likelihood difference relative to the best tree T0 (bar = 0). "
        "Trees with ΔLLH close to 0 are almost equally likely. "
        "A large gap between T0 and the rest indicates a single dominant topology.")
    _save(pdf, fig)


# ── p: posterior probability ───────────────────────────────────────────────────
def page_posterior(pdf, d, sample_id):
    fig, ax = _new_page("Posterior Probability & Sample Count per Tree", sample_id)
    n = d["n_trees"]
    x = np.arange(n)
    bars = ax.bar(x, d["prob"], color="#55A868", edgecolor="white", alpha=0.85,
                  label="Posterior probability")
    for bar, p in zip(bars, d["prob"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(d["prob"]) * 0.01,
                f"{p:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, d["count"], "D--", color="#555555", ms=7, lw=1.5, label="Sample count")
    ax2.set_ylabel("Times sampled", fontsize=9, color="#555555")
    ax2.tick_params(colors="#555555", labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n)], fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    _style(ax, "Posterior Probability & Sample Count",
           "Tree index", "Posterior probability")
    _caption(fig,
        "Posterior probability (bars) and number of times each tree was sampled (diamonds). "
        "A tree with probability ≈ 0.5 and sampled frequently is strongly supported. "
        "Low-probability trees may still be informative if ΔLLH is small.")
    _save(pdf, fig)


# ── p: φ bar chart (best tree) ────────────────────────────────────────────────
def page_phi(pdf, d, sample_id):
    phi_b   = d["phi"][0, :, 0]   # (K+1,)
    K       = d["K"]
    fig, ax = _new_page("Population Frequency φ per Node  (best tree)", sample_id)

    node_labels = ["Normal\n(root)"] + [f"C{k+1}" for k in range(K)]
    cols        = [_nc(n) for n in range(K + 1)]
    x           = np.arange(K + 1)

    bars = ax.bar(x, phi_b, color=cols, edgecolor="white", alpha=0.87, width=0.55)
    for bar, val in zip(bars, phi_b):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012, f"{val:.4f}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(node_labels, fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.axhline(1.0, color="#E74C3C", ls="--", lw=1.0, alpha=0.5)
    _style(ax, "φ — Population Frequency per Node  (fraction of ALL cells)",
           "Node", "φ")
    _caption(fig,
        "φ (phi) = fraction of ALL cells (tumour + normal) carrying the mutations in that node's cluster. "
        "The normal root is always φ = 1 by definition. "
        "φ of a parent node is always ≥ the sum of its children's φ (subclone constraint).")
    _save(pdf, fig)


# ── p: η bar chart (best tree) ────────────────────────────────────────────────
def page_eta(pdf, d, sample_id):
    if d["eta"] is None:
        return
    eta_b   = d["eta"][0, :, 0]
    K       = d["K"]
    fig, ax = _new_page("Clone-exclusive Frequency η per Node  (best tree)", sample_id)

    node_labels = ["Normal\n(root)"] + [f"C{k+1}" for k in range(K)]
    cols        = [_nc(n) for n in range(K + 1)]
    x           = np.arange(K + 1)

    bars = ax.bar(x, eta_b, color=cols, edgecolor="white", alpha=0.87, width=0.55)
    for bar, val in zip(bars, eta_b):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.010, f"{val:.4f}",
                ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(node_labels, fontsize=10)
    ax.set_ylim(0, max(eta_b) * 1.25)
    _style(ax, "η — Clone-exclusive Frequency  (fraction of cells in this clone only)",
           "Node", "η")
    _caption(fig,
        "η (eta) = fraction of ALL cells that belong exclusively to this clone "
        "(i.e. not counted in its descendants). η values sum to 1 across all nodes. "
        "A high η means this clone dominates the cellular landscape at this node.")
    _save(pdf, fig)


# ── p: φ uncertainty across trees ─────────────────────────────────────────────
def page_phi_stability(pdf, d, sample_id):
    if d["n_trees"] < 2:
        return
    phi   = d["phi"][:, :, 0]   # (n_trees, K+1)
    K     = d["K"]
    n     = d["n_trees"]
    fig, ax = _new_page("φ Stability across all Sampled Trees", sample_id)

    x = np.arange(n)
    for node in range(1, K + 1):
        ax.plot(x, phi[:, node], marker="o", ms=7, lw=2,
                color=_nc(node), label=f"C{node}")
        ax.fill_between(x,
                        phi[:, node] - phi[:, node].std(),
                        phi[:, node] + phi[:, node].std(),
                        color=_nc(node), alpha=0.10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i}" for i in range(n)], fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right")
    _style(ax, "φ per Cluster across all Trees  (T0 = best)",
           "Tree index", "φ (population frequency)")
    _caption(fig,
        "φ for each tumour cluster across all sampled trees. "
        "Flat lines indicate a stable frequency estimate robust to topology uncertainty. "
        "Large variation suggests the cluster frequency is tied to a specific topology assumption.")
    _save(pdf, fig)


# ── p: CCF bar (best tree) ────────────────────────────────────────────────────
def page_ccf(pdf, d, sample_id):
    phi_b   = d["phi"][0, :, 0]
    K       = d["K"]
    clusters = d["clusters"]
    fig, ax = _new_page("Estimated Cancer Cell Fraction (CCF)  (best tree)", sample_id)

    cls_labels = [f"C{k+1}" for k in range(K)]
    ccfs       = [min(float(phi_b[k + 1]) * 2, 1.0) for k in range(K)]
    cols       = [_nc(k + 1) for k in range(K)]
    x          = np.arange(K)

    bars = ax.bar(x, ccfs, color=cols, edgecolor="white", alpha=0.87, width=0.55)
    ax.axhline(1.0, color="#E74C3C", ls="--", lw=1.2, label="CCF = 1.0  (clonal)")
    ax.axhline(0.5, color="#F39C12", ls=":",  lw=1.0, label="CCF = 0.5")
    for bar, ccf, k in zip(bars, ccfs, range(K)):
        n_m = len(clusters[k]) if k < len(clusters) else "?"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"CCF≈{ccf:.3f}\n(n={n_m})",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(cls_labels, fontsize=10)
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=9)
    _style(ax, "CCF = min(2φ, 1.0) per Cluster  (diploid assumption)",
           "Cluster", "Cancer Cell Fraction (CCF)")
    _caption(fig,
        "CCF (Cancer Cell Fraction) = estimated proportion of tumour cells carrying a cluster's mutations. "
        "Computed as CCF ≈ 2φ, valid under diploid heterozygous copy-neutral loci. "
        "Clusters with CCF ≈ 1 are clonal (founding event); lower-CCF clusters are subclonal.")
    _save(pdf, fig)


# ── p: tree interpretation monospace text ─────────────────────────────────────
def page_interpretation(pdf, d, sample_id):
    phi_b   = d["phi"][0]
    eta_b   = d["eta"][0] if d["eta"] is not None else None
    K       = d["K"]
    struct0 = d["struct"][0]
    newick0 = d["newick"][0]
    clusters = d["clusters"]

    fig, ax = _new_page("Best Tree — Annotated Interpretation", sample_id)
    ax.axis("off")

    children = _build_children(struct0)
    lines = [
        "CLONAL ARCHITECTURE  (best tree)",
        f"Newick: {newick0}",
        "─" * 55,
        "",
        "Node 0  [Normal / germline root]",
        f"  φ = 1.000  (all cells by definition)",
        f"  Children: {children.get(0, [])}",
        "",
    ]
    for k in range(K):
        node  = k + 1
        phi_v = float(phi_b[node, 0])
        eta_v = float(eta_b[node, 0]) if eta_b is not None else None
        n_mut = len(clusters[k]) if k < len(clusters) else "?"
        par   = int(struct0[k])
        ch    = children.get(node, [])
        ccf   = min(phi_v * 2, 1.0)
        bio   = ("CLONAL — founding tumour event" if phi_v >= 0.40
                 else "SUBCLONAL — acquired in a cell subset" if phi_v >= 0.15
                 else "RARE SUBCLONE — very small cell fraction")

        lines += [
            f"Node {node}  [Cluster C{node}]",
            f"  Mutations : {n_mut}",
            f"  φ         = {phi_v:.4f}   →   CCF ≈ {ccf:.3f}",
        ]
        if eta_v is not None:
            lines.append(f"  η         = {eta_v:.4f}  (exclusive fraction)")
        lines += [
            f"  Parent    : node {par}",
            f"  Children  : {ch}",
            f"  Class     : {bio}",
            "",
        ]
    lines += [
        "─" * 55,
        "φ = population frequency (fraction of ALL cells)",
        "η = clone-exclusive frequency (only this clone, not descendants)",
        "CCF ≈ 2φ  (diploid heterozygous, 100% purity assumption)",
    ]

    ax.text(0.02, 0.97, "\n".join(lines),
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9, color="#222222",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.7",
                      facecolor="#F8F9FA", edgecolor="#AED6F1", alpha=0.9))

    _caption(fig,
        "Plain-language summary of the best tree's clonal architecture. "
        "Each cluster node is annotated with its population frequency (φ), "
        "clone-exclusive fraction (η), estimated CCF, and evolutionary classification.")
    _save(pdf, fig)


# ── p: VAF histogram — one page per cluster ───────────────────────────────────
def pages_vaf_per_cluster(pdf, d, sample_id):
    ssm      = d["ssm"]
    K        = d["K"]
    phi_best = d["phi"][0]
    clusters = d["clusters"]

    if ssm is None:
        return

    bins = np.linspace(0, 1, 41)

    for k in range(K):
        node    = k + 1
        col     = _nc(node)
        sub     = ssm[ssm["cluster"] == k]
        phi_v   = float(phi_best[node, 0])
        vaf_exp = phi_v / 2.0
        n_cl    = len(sub)

        fig, ax = _new_page(
            f"VAF Distribution — Cluster C{node}  (φ={phi_v:.3f}  n={n_cl})",
            sample_id)

        if n_cl > 0:
            ax.hist(sub["vaf"], bins=bins, color=col, alpha=0.78, edgecolor="white")
        ax.axvline(vaf_exp, color="black", lw=2, ls="--",
                   label=f"Expected φ/2 = {vaf_exp:.3f}")
        if n_cl > 0:
            med = sub["vaf"].median()
            ax.axvline(med, color=col, lw=1.8, ls=":",
                       label=f"Observed median = {med:.3f}")
        ax.legend(fontsize=9)
        _style(ax, f"VAF Distribution — Cluster C{node}",
               "Variant Allele Frequency (VAF)", "Mutations")

        _caption(fig,
            f"VAF histogram for all {n_cl} mutations assigned to cluster C{node} "
            f"(φ = {phi_v:.4f}). "
            "Dashed black = expected VAF = φ/2 under diploid heterozygous loci and full purity. "
            "Coloured dotted = observed median. "
            "Deviation may indicate copy-number gains or impure tumour purity.")
        _save(pdf, fig)


# ── p: read depth — one page per cluster ──────────────────────────────────────
def pages_depth_per_cluster(pdf, d, sample_id):
    ssm      = d["ssm"]
    K        = d["K"]
    clusters = d["clusters"]

    if ssm is None:
        return

    all_depth = ssm["total_reads"].dropna()
    p99       = np.percentile(all_depth, 99)
    overall_med = all_depth.median()

    for k in range(K):
        node = k + 1
        col  = _nc(node)
        sub  = ssm[ssm["cluster"] == k]["total_reads"].dropna()

        fig, ax = _new_page(
            f"Read Depth — Cluster C{node}  (n={len(sub)})", sample_id)

        if len(sub) > 0:
            ax.hist(sub.clip(upper=p99), bins=50, color=col,
                    alpha=0.78, edgecolor="white")
            ax.axvline(sub.median(), color="black", lw=2, ls="--",
                       label=f"Cluster median = {sub.median():.0f}×")
        ax.axvline(overall_med, color="#AAAAAA", lw=1.5, ls=":",
                   label=f"Overall median = {overall_med:.0f}×")
        ax.legend(fontsize=9)
        _style(ax, f"Read Depth Distribution — Cluster C{node}",
               "Total reads (×)", "Mutations")

        _caption(fig,
            f"Total read depth (total_reads from SSM) for mutations in cluster C{node}. "
            "Depth should be comparable across clusters. "
            "Systematically lower depth may reduce φ estimation accuracy for this cluster. "
            "Clipped at 99th percentile for display.")
        _save(pdf, fig)


# ── p: chromosome distribution (absolute) ─────────────────────────────────────
def page_chrom_absolute(pdf, d, sample_id):
    ssm      = d["ssm"]
    K        = d["K"]
    clusters = d["clusters"]

    if ssm is None or ssm["chrom"].isna().all():
        return

    ssm2        = ssm.dropna(subset=["cluster", "chrom"])
    present     = [c for c in CHR_ORDER if c in ssm2["chrom"].values]
    other       = sorted(set(ssm2["chrom"].unique()) - set(CHR_ORDER))
    chrom_order = present + other
    x           = np.arange(len(chrom_order))

    fig, ax = _new_page("Chromosomal Distribution — Absolute Counts", sample_id)
    ax.set_position([0.08, PLOT_BOTTOM, 0.87, PLOT_TOP - PLOT_BOTTOM])

    bottom = np.zeros(len(chrom_order))
    for k in range(K):
        counts = (ssm2[ssm2["cluster"] == k]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        ax.bar(x, counts, bottom=bottom, label=f"C{k+1}",
               color=_nc(k + 1), edgecolor="white", linewidth=0.3)
        bottom += counts

    ax.set_xticks(x)
    ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7.5)
    ax.legend(fontsize=9, loc="upper right")
    _style(ax, "Mutations per Chromosome (stacked by cluster)", "Chromosome", "Count")

    _caption(fig,
        "Absolute mutation counts per chromosome, stacked by Orchard cluster assignment. "
        "Each chromosome's total bar height = number of mutations on that chromosome.")
    _save(pdf, fig)


# ── p: chromosome distribution (100% stacked) ─────────────────────────────────
def page_chrom_fraction(pdf, d, sample_id):
    ssm      = d["ssm"]
    K        = d["K"]

    if ssm is None or ssm["chrom"].isna().all():
        return

    ssm2        = ssm.dropna(subset=["cluster", "chrom"])
    present     = [c for c in CHR_ORDER if c in ssm2["chrom"].values]
    other       = sorted(set(ssm2["chrom"].unique()) - set(CHR_ORDER))
    chrom_order = present + other
    x           = np.arange(len(chrom_order))
    totals      = ssm2.groupby("chrom").size().reindex(chrom_order, fill_value=0).values.astype(float)

    fig, ax = _new_page("Chromosomal Distribution — Cluster Composition (100% stacked)", sample_id)
    ax.set_position([0.08, PLOT_BOTTOM, 0.87, PLOT_TOP - PLOT_BOTTOM])

    bottom = np.zeros(len(chrom_order))
    for k in range(K):
        counts = (ssm2[ssm2["cluster"] == k]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        fracs  = np.where(totals > 0, counts / totals, 0)
        ax.bar(x, fracs, bottom=bottom, label=f"C{k+1}",
               color=_nc(k + 1), edgecolor="white", linewidth=0.3)
        bottom += fracs

    ax.set_xticks(x)
    ax.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7.5)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")
    _style(ax, "Cluster Composition per Chromosome (100% stacked)", "Chromosome", "Fraction")

    _caption(fig,
        "Fraction of mutations on each chromosome belonging to each cluster. "
        "Uniform composition across chromosomes supports a genuine clonal signal. "
        "Strong enrichment of one cluster on a specific chromosome may indicate "
        "residual copy-number artefacts in the clustering.")
    _save(pdf, fig)


# ── p: genome-wide VAF landscape ──────────────────────────────────────────────
def page_genome_vaf(pdf, d, sample_id):
    ssm      = d["ssm"]
    K        = d["K"]
    phi_best = d["phi"][0]

    if ssm is None or ssm["chrom"].isna().all():
        return

    present = [c for c in CHR_ORDER if c in ssm["chrom"].dropna().values]
    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum
        cum += CHR_LENGTHS.get(ch, 100e6)

    ssm2     = ssm[ssm["chrom"].isin(present)].copy()
    ssm2["gx"] = ssm2["chrom"].map(offsets).fillna(0) + ssm2["pos"].fillna(0)

    fig, ax = _new_page("Genome-wide VAF Landscape", sample_id)

    for k in range(K):
        sub = ssm2[ssm2["cluster"] == k]
        ax.scatter(sub["gx"], sub["vaf"], color=_nc(k + 1),
                   s=7, alpha=0.5, edgecolors="none", label=f"C{k+1}")
        ax.axhline(float(phi_best[k + 1, 0]) / 2.0,
                   color=_nc(k + 1), lw=1.0, ls="--", alpha=0.6)
    unassigned = ssm2[ssm2["cluster"].isna()]
    if len(unassigned):
        ax.scatter(unassigned["gx"], unassigned["vaf"],
                   color="#CCCCCC", s=4, alpha=0.3, edgecolors="none",
                   label="unassigned")

    for ch in present:
        ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        ax.text(mid, -0.06, ch.replace("chr", ""),
                ha="center", fontsize=6, color="#555555",
                transform=ax.get_xaxis_transform())

    ax.set_xlim(0, cum)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper right", ncol=2, markerscale=1.8)
    _style(ax, "VAF across the Genome (coloured by Orchard cluster)",
           "Genomic position (hg38)", "Variant Allele Frequency (VAF)")

    _caption(fig,
        "Genome-wide scatter of all mutations coloured by Orchard cluster. "
        "Dashed horizontal lines = expected VAF = φ/2 per cluster (diploid het). "
        "Systematic VAF deviations from the expected line in a chromosomal region "
        "may indicate focal copy-number events not modelled by Orchard.")
    _save(pdf, fig)


# ── p: genome-wide depth landscape ────────────────────────────────────────────
def page_genome_depth(pdf, d, sample_id):
    ssm = d["ssm"]
    if ssm is None or ssm["chrom"].isna().all():
        return

    present = [c for c in CHR_ORDER if c in ssm["chrom"].dropna().values]
    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum
        cum += CHR_LENGTHS.get(ch, 100e6)

    ssm2     = ssm[ssm["chrom"].isin(present)].copy()
    ssm2["gx"] = ssm2["chrom"].map(offsets).fillna(0) + ssm2["pos"].fillna(0)

    fig, ax = _new_page("Genome-wide Read Depth Landscape", sample_id)

    for k in range(d["K"]):
        sub = ssm2[ssm2["cluster"] == k]
        ax.scatter(sub["gx"], sub["total_reads"], color=_nc(k + 1),
                   s=6, alpha=0.45, edgecolors="none", label=f"C{k+1}")

    for ch in present:
        ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6) / 2
        ax.text(mid, -0.04, ch.replace("chr", ""),
                ha="center", fontsize=6, color="#555555",
                transform=ax.get_xaxis_transform())

    overall_med = ssm2["total_reads"].median()
    ax.axhline(overall_med, color="black", ls="--", lw=1.2,
               label=f"Median = {overall_med:.0f}×")
    ax.set_xlim(0, cum)
    ax.legend(fontsize=8, loc="upper right", ncol=2, markerscale=1.8)
    _style(ax, "Read Depth across the Genome (coloured by cluster)",
           "Genomic position (hg38)", "Total reads (×)")

    _caption(fig,
        "Read depth per mutation coloured by cluster assignment. "
        "Depth should be roughly uniform across the genome. "
        "Depth valleys may indicate poor mappability regions or structural variants.")
    _save(pdf, fig)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Orchard phylogenetic report — one plot per DIN A4 page."
    )
    parser.add_argument("--npz",    "-n", required=True)
    parser.add_argument("--params", "-p", default=None)
    parser.add_argument("--ssm",    "-m", default=None)
    parser.add_argument("--sample", "-s", default="sample")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    sid     = args.sample
    out_pdf = Path(args.output) if args.output else Path(f"{sid}_orchard_report.pdf")

    print(f"[orchard_report] Sample: {sid}")
    d = load_data(args.npz, args.params, args.ssm)
    print(f"[orchard_report] {d['n_trees']} trees  K={d['K']}  S={d['S']}")
    print(f"[orchard_report] cluster sizes: {[len(c) for c in d['clusters']]}")
    print(f"[orchard_report] best LLH={d['llh'][0]:.4f}  newick={d['newick'][0]}")

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.5,
    })

    print(f"[orchard_report] Writing: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        info = pdf.infodict()
        info["Title"]  = f"Orchard Report — {sid}"
        info["Author"] = "orchard_report.py"

        page_summary_table(pdf, d, sid)       # p1  — run statistics table
        page_best_tree(pdf, d, sid)           # p2  — best tree annotated
        pages_all_trees(pdf, d, sid)          # p3+ — one page per tree
        page_llh(pdf, d, sid)                 # p?  — ΔLLH bar
        page_posterior(pdf, d, sid)           # p?  — posterior prob + count
        page_phi(pdf, d, sid)                 # p?  — φ bar chart
        page_eta(pdf, d, sid)                 # p?  — η bar chart (if available)
        page_phi_stability(pdf, d, sid)       # p?  — φ across trees
        page_ccf(pdf, d, sid)                 # p?  — CCF bar
        page_interpretation(pdf, d, sid)      # p?  — monospace interpretation
        pages_vaf_per_cluster(pdf, d, sid)    # p?+ — one VAF page per cluster
        pages_depth_per_cluster(pdf, d, sid)  # p?+ — one depth page per cluster
        page_chrom_absolute(pdf, d, sid)      # p?  — chrom absolute
        page_chrom_fraction(pdf, d, sid)      # p?  — chrom 100% stacked
        page_genome_vaf(pdf, d, sid)          # p?  — genome VAF
        page_genome_depth(pdf, d, sid)        # p?  — genome depth

    print(f"[orchard_report] Done — {out_pdf}")


if __name__ == "__main__":
    main()