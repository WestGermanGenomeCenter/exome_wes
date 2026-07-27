#!/usr/bin/env python3
"""
compare_clonal_tools.py — Cross-tool comparison of clonal reconstruction output.

Compares whichever of VIBER, PyClone6, Orchard, PhylogicNDT, and muttime
actually ran for a sample, and reports where they agree and disagree on:
cluster counts, per-mutation CCF, clonal/subclonal calls, and (for muttime)
mutation timing. Only tools with usable output are included -- nothing is
required, nothing crashes if a tool is missing or partially failed.

Genome-build agnostic: chromosome names are matched by stripping any
'chr'/'Chr'/'CHR' prefix before comparison, so this works regardless of
whether the underlying reference used UCSC-style ('chr1') or Ensembl-style
('1') contig names (hg38, hs1/T2T-CHM13, mm10/mm39, etc.). No chromosome
length table is used anywhere, so no genome build needs to be known.

This script does NOT compute a composite "agreement score" or rank tools.
It shows raw counts, percentages, and correlations so you can judge for
yourself. Sections are omitted (not guessed) when the underlying data
isn't available for a given tool.

Usage:
    python compare_clonal_tools.py \\
        --sample-dir  results/synovial_sarcoma_WXS_2025_sample1 \\
        --sample      synovial_sarcoma_WXS_2025_sample1 \\
        [--clonal-threshold 0.7] \\
        [--output-prefix synovial_sarcoma_WXS_2025_sample1_compare]

    # or point to each tool's directory explicitly:
    python compare_clonal_tools.py \\
        --sample synovial_sarcoma_WXS_2025_sample1 \\
        --viber-dir      results/.../viber \\
        --pyclone6-dir   results/.../pyclone6 \\
        --orchard-dir    results/.../orchard \\
        --phylogic-dir   results/.../phylogic \\
        --muttime-dir    results/.../muttime \\
        --facets-dir     results/.../facets \\
        --cnaqc-dir      results/.../cnaqc

Expected files per tool (auto-detected under --sample-dir/<toolname>/ if a
tool-specific --*-dir is not given):
    viber/       mutations.tsv (mutation_id, cluster, successes, trials)
                 parameters.tsv (cluster, pi, theta)
    pyclone6/    <sample>_pyclone6_results.tsv
    orchard/     <sample>.orchard.npz, <sample>.params.json, <sample>.ssm
    phylogic/    <sample>.mut_ccfs.txt, <sample>.cluster_ccfs.txt
    muttime/     <sample>_mutations.tsv
    facets/      <sample>_facets_qc.txt   (purity cross-check only)
    cnaqc/       <sample>_cnaqc_qc.txt    (purity cross-check only)

Output files (written before the PDF):
    <prefix>_per_mutation_comparison.tsv
    <prefix>_per_cluster_summary.tsv
    <prefix>_upstream_purity.tsv
    <prefix>_comparison_report.pdf

Conda deps (all conda-forge):
    conda install -c conda-forge pandas numpy matplotlib scipy
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
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# ── fixed per-tool colours (consistent across every sample's report, so you
#    can flip between multiple reports and immediately recognise each tool) ──
TOOL_COLOURS = {
    "VIBER":       "#4C72B0",
    "PyClone6":    "#DD8452",
    "Orchard":     "#55A868",
    "PhylogicNDT": "#C44E52",
    "muttime":     "#8172B2",
}
# whether a tool represents a real clustering (used for the cluster-count page)
TOOL_IS_CLUSTERING = {
    "VIBER": True, "PyClone6": True, "Orchard": True,
    "PhylogicNDT": True, "muttime": False,
}
TOOL_ORDER = ["VIBER", "PyClone6", "Orchard", "PhylogicNDT", "muttime"]

def warn(msg):  print(f"  [WARN] {msg}", file=sys.stderr)
def info(msg):  print(f"  [INFO] {msg}", file=sys.stderr)


# ── genome-agnostic helpers ─────────────────────────────────────────────────────

def _normalise_chrom(s):
    """Strip any leading chr/Chr/CHR prefix and lowercase. Works regardless
    of reference genome naming convention (UCSC 'chr1' vs Ensembl '1')."""
    s = str(s).strip()
    if s[:3].lower() == "chr":
        s = s[3:]
    return s.lower()


def _parse_mutation_id(mid):
    """'chr1:12345:A>T' -> ('chr1','12345','A','T'). Tolerant of indels
    (multi-character ref/alt). Returns (None,None,None,None) on failure."""
    try:
        chrom, pos, refalt = str(mid).split(":", 2)
        ref, alt = refalt.split(">", 1)
        return chrom, pos, ref, alt
    except Exception:
        return None, None, None, None


def _match_key_from_parts(chrom, pos, ref, alt):
    c = _normalise_chrom(chrom)
    try:
        p = str(int(float(pos)))
    except Exception:
        p = str(pos).strip()
    r = str(ref).strip().upper()
    a = str(alt).strip().upper()
    return f"{c}:{p}:{r}>{a}"


def _match_key_from_id(mid):
    chrom, pos, ref, alt = _parse_mutation_id(mid)
    if chrom is None:
        return None
    return _match_key_from_parts(chrom, pos, ref, alt)


# ── purity helper (handles both key-value and single-scalar file formats) ──────

def _read_purity(path, key_candidates):
    """
    Reads a purity value from either:
      - a 2-column key\\tvalue file (no header), e.g. 'purity_used\\t0.84'
      - a single-scalar file, e.g. just '0.84'
    Returns float or None.
    """
    if path is None or not Path(path).exists():
        return None
    try:
        with open(path) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]
        if not lines:
            return None
        # try key-value format first
        for line in lines:
            fields = line.split("\t")
            if len(fields) >= 2 and fields[0].strip().lower() in \
                    [k.lower() for k in key_candidates]:
                try:
                    return float(fields[1])
                except ValueError:
                    continue
        # fall back to treating the first line as a bare scalar
        try:
            return float(lines[0].split("\t")[0])
        except ValueError:
            return None
    except Exception as e:
        warn(f"Could not read purity from {path}: {e}")
        return None


def load_upstream_purity(sample, facets_dir, cnaqc_dir, phylogic_dir, pyclone6_dir):
    """Collect purity from every upstream source that's available, for a
    quick cross-check (differing input purity is a common, concrete reason
    two clonal tools might disagree)."""
    rows = []

    if facets_dir:
        p = _read_purity(Path(facets_dir) / f"{sample}_facets_qc.txt", ["purity"])
        if p is not None:
            rows.append(("FACETS (facets_qc.txt)", p))

    if cnaqc_dir:
        p = _read_purity(Path(cnaqc_dir) / f"{sample}_cnaqc_qc.txt", ["purity_used", "purity"])
        if p is not None:
            rows.append(("CNAqc (cnaqc_qc.txt)", p))

    if phylogic_dir:
        p = _read_purity(Path(phylogic_dir) / f"{sample}_purity.txt", ["purity"])
        if p is not None:
            rows.append(("PhylogicNDT (purity.txt input)", p))

    if pyclone6_dir:
        inp_path = Path(pyclone6_dir) / f"{sample}_pyclone6_input.tsv"
        if inp_path.exists():
            try:
                inp = pd.read_csv(inp_path, sep="\t")
                if "tumour_content" in inp.columns:
                    val = pd.to_numeric(inp["tumour_content"], errors="coerce").median()
                    if pd.notna(val):
                        rows.append(("PyClone6 (input tumour_content, median)", float(val)))
            except Exception as e:
                warn(f"Could not read PyClone6 input purity: {e}")

    return pd.DataFrame(rows, columns=["source", "purity"]) if rows else None


# ── tool adapters ────────────────────────────────────────────────────────────────
# Each adapter returns (mut_df, cluster_df) or (None, None).
# mut_df columns:     match_key, mutation_id, tool, cluster_label, ccf
# cluster_df columns: tool, cluster_label, n_mutations, ccf

def load_viber(viber_dir, sample):
    mut_p = Path(viber_dir) / "mutations.tsv"
    par_p = Path(viber_dir) / "parameters.tsv"
    if not mut_p.exists() or not par_p.exists():
        info(f"VIBER: mutations.tsv or parameters.tsv not found under {viber_dir} — skipping")
        return None, None
    try:
        mut = pd.read_csv(mut_p, sep="\t")
        par = pd.read_csv(par_p, sep="\t")
    except Exception as e:
        warn(f"VIBER: failed to read files: {e}")
        return None, None

    if "cluster" not in mut.columns or "cluster" not in par.columns or "theta" not in par.columns:
        warn("VIBER: mutations.tsv/parameters.tsv missing required columns — skipping")
        return None, None

    if "mutation_id" not in mut.columns:
        # fall back to row-position join with *_viber_input.tsv if present
        input_p = Path(viber_dir) / f"{sample}_viber_input.tsv"
        if input_p.exists():
            try:
                inp = pd.read_csv(input_p, sep="\t")
                if len(inp) == len(mut) and "mutation_id" in inp.columns:
                    mut = mut.reset_index(drop=True)
                    mut["mutation_id"] = inp["mutation_id"].values
                else:
                    warn("VIBER: viber_input.tsv row count mismatch — mutation-level "
                         "comparison unavailable for VIBER")
                    mut["mutation_id"] = pd.NA
            except Exception as e:
                warn(f"VIBER: could not read viber_input.tsv fallback: {e}")
                mut["mutation_id"] = pd.NA
        else:
            warn("VIBER: no mutation_id column and no viber_input.tsv fallback found — "
                 "mutation-level comparison unavailable for VIBER")
            mut["mutation_id"] = pd.NA

    par["theta"] = pd.to_numeric(par["theta"], errors="coerce")
    par["pi"]    = pd.to_numeric(par.get("pi", np.nan), errors="coerce")
    par_idx = par.set_index("cluster")

    def theta_to_ccf(theta):
        if pd.isna(theta):
            return np.nan
        return min(theta * 2, 1.0)   # theta ~ VAF peak; CCF ~ 2*VAF (diploid)

    mut["ccf"] = mut["cluster"].map(
        lambda c: theta_to_ccf(par_idx.loc[c, "theta"]) if c in par_idx.index else np.nan)
    mut["cluster_label"] = mut["cluster"].astype(str)
    mut["tool"] = "VIBER"
    mut["match_key"] = mut["mutation_id"].apply(_match_key_from_id)

    n_ok = mut["match_key"].notna().sum()
    info(f"VIBER: {len(mut)} mutations loaded, {n_ok} with usable mutation IDs, "
         f"{par['cluster'].nunique()} clusters")

    cluster_df = par.copy()
    cluster_df["tool"] = "VIBER"
    cluster_df["cluster_label"] = cluster_df["cluster"].astype(str)
    cluster_df["ccf"] = cluster_df["theta"].apply(theta_to_ccf)
    counts = mut["cluster"].value_counts()
    cluster_df["n_mutations"] = cluster_df["cluster"].map(counts).fillna(0).astype(int)

    return (mut[["match_key","mutation_id","tool","cluster_label","ccf"]],
            cluster_df[["tool","cluster_label","n_mutations","ccf"]])


def load_pyclone6(pyclone6_dir, sample):
    p = Path(pyclone6_dir) / f"{sample}_pyclone6_results.tsv"
    if not p.exists():
        info(f"PyClone6: results file not found ({p}) — skipping")
        return None, None
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception as e:
        warn(f"PyClone6: failed to read results: {e}")
        return None, None

    required = {"mutation_id", "cluster_id", "cellular_prevalence"}
    missing = required - set(df.columns)
    if missing:
        warn(f"PyClone6: missing columns {missing} — skipping")
        return None, None

    df["ccf"] = pd.to_numeric(df["cellular_prevalence"], errors="coerce")
    df["cluster_label"] = "C" + df["cluster_id"].astype(str)
    df["tool"] = "PyClone6"
    df["match_key"] = df["mutation_id"].apply(_match_key_from_id)

    n_ok = df["match_key"].notna().sum()
    info(f"PyClone6: {len(df)} mutations loaded, {n_ok} with usable mutation IDs, "
         f"{df['cluster_id'].nunique()} clusters")

    cluster_df = (df.groupby("cluster_label")
                    .agg(n_mutations=("mutation_id","count"), ccf=("ccf","median"))
                    .reset_index())
    cluster_df["tool"] = "PyClone6"

    return (df[["match_key","mutation_id","tool","cluster_label","ccf"]],
            cluster_df[["tool","cluster_label","n_mutations","ccf"]])


def load_orchard(orchard_dir, sample):
    npz_p = Path(orchard_dir) / f"{sample}.orchard.npz"
    par_p = Path(orchard_dir) / f"{sample}.params.json"
    ssm_p = Path(orchard_dir) / f"{sample}.ssm"
    if not (npz_p.exists() and par_p.exists() and ssm_p.exists()):
        info(f"Orchard: required files not all found under {orchard_dir} — skipping")
        return None, None
    try:
        npz = np.load(npz_p, allow_pickle=True)
        ssm = pd.read_csv(ssm_p, sep="\t")
        with open(par_p) as f:
            params = json.load(f)
    except Exception as e:
        warn(f"Orchard: failed to load files: {e}")
        return None, None

    clusters = params.get("clusters", [])
    if not clusters:
        warn("Orchard: no clusters found in params.json — skipping")
        return None, None
    if "phi" not in npz.files:
        warn("Orchard: NPZ has no 'phi' array — skipping")
        return None, None

    phi_best = npz["phi"][0, :, 0]   # (K+1,) best tree, first sample
    id2mutid = dict(zip(ssm["id"], ssm["name"]))

    rows = []
    for k, cl in enumerate(clusters):
        node = k + 1
        if node >= len(phi_best):
            continue
        # CCF ≈ φ directly: var_read_prob=0.5 fixed in SSM already encodes
        # the diploid-het VAF↔φ relationship (see orchard_report.py note).
        ccf = min(float(phi_best[node]), 1.0)
        for sid in cl:
            mutation_id = id2mutid.get(sid)
            if mutation_id is None:
                continue
            rows.append(dict(mutation_id=mutation_id, cluster_label=f"C{node}", ccf=ccf))

    if not rows:
        warn("Orchard: no mutations could be mapped from clusters to SSM IDs — skipping")
        return None, None

    mut = pd.DataFrame(rows)
    mut["tool"] = "Orchard"
    mut["match_key"] = mut["mutation_id"].apply(_match_key_from_id)

    n_ok = mut["match_key"].notna().sum()
    info(f"Orchard: {len(mut)} mutations loaded, {n_ok} with usable mutation IDs, "
         f"{len(clusters)} clusters (best tree)")

    cluster_df = (mut.groupby("cluster_label")
                    .agg(n_mutations=("mutation_id","count"), ccf=("ccf","first"))
                    .reset_index())
    cluster_df["tool"] = "Orchard"

    return (mut[["match_key","mutation_id","tool","cluster_label","ccf"]],
            cluster_df[["tool","cluster_label","n_mutations","ccf"]])


# known PhylogicNDT header tokens that get concatenated without a tab in
# some versions -- observed empirically in real pipeline output.
_PHY_KNOWN_MERGES = {
    "Protein_changeVariant_Classification": ["Protein_change", "Variant_Classification"],
    "clust_ccf_meanclust_ccf_CI_low":       ["clust_ccf_mean", "clust_ccf_CI_low"],
}
# fallback column positions from the confirmed real header layout, used only
# if name-based repair fails entirely
_PHY_FALLBACK_POS = {
    "Chromosome": 4, "Start_position": 5, "Reference_Allele": 6,
    "Tumor_Seq_Allele": 7, "Cluster_Assignment": 13, "preDP_ccf_mean": 16,
}


def _read_phylogic_mut_ccfs(path):
    """
    Robustly reads PhylogicNDT's mut_ccfs.txt. Known issue: some PhylogicNDT
    versions write a header row with two adjacent column names concatenated
    (missing tab) while data rows remain correctly tab-separated -- this
    silently breaks naive pd.read_csv column alignment. We repair known
    merges, and fall back to position-anchored parsing of the reliably
    unaffected early columns if repair isn't possible.
    Returns (DataFrame with columns needed, ccf_source_used) or (None, None).
    """
    try:
        with open(path) as f:
            header_line = f.readline()
            first_data_line = f.readline()
    except Exception as e:
        warn(f"PhylogicNDT: could not open mut_ccfs.txt: {e}")
        return None, None

    header_tokens = header_line.rstrip("\n").split("\t")
    data_tokens = first_data_line.rstrip("\n").split("\t") if first_data_line else []

    repaired = []
    for tok in header_tokens:
        repaired.extend(_PHY_KNOWN_MERGES.get(tok, [tok]))

    header_ok = (len(data_tokens) == 0) or (len(repaired) == len(data_tokens))
    if not header_ok:
        warn(f"PhylogicNDT mut_ccfs.txt: header field count ({len(repaired)}) still doesn't "
             f"match data field count ({len(data_tokens)}) after known-merge repair. "
             f"Falling back to positional parsing of core columns only.")

    needed = ["Chromosome", "Start_position", "Reference_Allele", "Tumor_Seq_Allele",
              "Cluster_Assignment", "preDP_ccf_mean", "clust_ccf_mean"]

    if header_ok:
        name_to_idx = {name: i for i, name in enumerate(repaired)}
        core_required = [n for n in needed if n != "clust_ccf_mean"]
        missing = [n for n in core_required if n not in name_to_idx]
        if missing:
            warn(f"PhylogicNDT mut_ccfs.txt missing required columns: {missing} — skipping tool")
            return None, None
        has_clust = "clust_ccf_mean" in name_to_idx
        if not has_clust:
            info("PhylogicNDT: clust_ccf_mean not present/recoverable — using preDP_ccf_mean "
                 "instead (pre-clustering per-mutation estimate, noisier)")
        idx_map = {n: name_to_idx[n] for n in needed if n in name_to_idx}
    else:
        if len(data_tokens) <= max(_PHY_FALLBACK_POS.values()):
            warn("PhylogicNDT mut_ccfs.txt: too few columns even for positional fallback — "
                 "skipping tool")
            return None, None
        idx_map = dict(_PHY_FALLBACK_POS)
        has_clust = False
        info("PhylogicNDT: using positional fallback parsing (header alignment could not "
             "be repaired) — only core columns recovered, clust_ccf_mean unavailable, "
             "using preDP_ccf_mean instead")

    rows = []
    n_bad_rows = 0
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(idx_map.values()):
                n_bad_rows += 1
                continue
            rows.append({name: fields[idx] for name, idx in idx_map.items()})

    if n_bad_rows > 0:
        warn(f"PhylogicNDT mut_ccfs.txt: skipped {n_bad_rows} malformed row(s)")

    if not rows:
        warn("PhylogicNDT mut_ccfs.txt: no usable data rows found — skipping tool")
        return None, None

    df = pd.DataFrame(rows)
    ccf_col = "clust_ccf_mean" if has_clust else "preDP_ccf_mean"
    return df, ccf_col


def load_phylogic(phylogic_dir, sample):
    mut_p = Path(phylogic_dir) / f"{sample}.mut_ccfs.txt"
    cl_p  = Path(phylogic_dir) / f"{sample}.cluster_ccfs.txt"
    if not mut_p.exists():
        info(f"PhylogicNDT: mut_ccfs.txt not found ({mut_p}) — skipping")
        return None, None

    df, ccf_col = _read_phylogic_mut_ccfs(mut_p)
    if df is None:
        return None, None

    df["ccf"] = pd.to_numeric(df[ccf_col], errors="coerce")
    df["chrom_norm"] = df["Chromosome"].apply(_normalise_chrom)
    df["match_key"] = df.apply(
        lambda r: _match_key_from_parts(
            r["Chromosome"], r["Start_position"],
            r["Reference_Allele"], r["Tumor_Seq_Allele"]),
        axis=1)
    df["mutation_id"] = (df["chrom_norm"] + ":" + df["Start_position"].astype(str) + ":" +
                         df["Reference_Allele"].astype(str) + ">" + df["Tumor_Seq_Allele"].astype(str))
    df["cluster_label"] = "C" + df["Cluster_Assignment"].astype(str)
    df["tool"] = "PhylogicNDT"

    mut = df[["match_key","mutation_id","tool","cluster_label","ccf"]].dropna(subset=["ccf"])
    info(f"PhylogicNDT: {len(mut)} mutations loaded (CCF from {ccf_col}), "
         f"{mut['cluster_label'].nunique()} clusters")

    cluster_df = None
    if cl_p.exists():
        try:
            cl = pd.read_csv(cl_p, sep="\t")
            if {"Cluster_ID", "postDP_ccf_mean"}.issubset(cl.columns):
                cl["cluster_label"] = "C" + cl["Cluster_ID"].astype(str)
                cl["ccf"] = pd.to_numeric(cl["postDP_ccf_mean"], errors="coerce")
                cl["tool"] = "PhylogicNDT"
                n_mut_map = mut["cluster_label"].value_counts()
                cl["n_mutations"] = cl["cluster_label"].map(n_mut_map).fillna(0).astype(int)
                cluster_df = cl[["tool","cluster_label","n_mutations","ccf"]]
            else:
                warn("PhylogicNDT: cluster_ccfs.txt missing expected columns — "
                     "deriving cluster summary from mut_ccfs.txt instead")
        except Exception as e:
            warn(f"PhylogicNDT: could not read cluster_ccfs.txt: {e}")

    if cluster_df is None:
        cluster_df = (mut.groupby("cluster_label")
                        .agg(n_mutations=("mutation_id","count"), ccf=("ccf","median"))
                        .reset_index())
        cluster_df["tool"] = "PhylogicNDT"
        cluster_df = cluster_df[["tool","cluster_label","n_mutations","ccf"]]

    return mut, cluster_df


def load_muttime(muttime_dir, sample):
    p = Path(muttime_dir) / f"{sample}_mutations.tsv"
    if not p.exists():
        info(f"muttime: mutations file not found ({p}) — skipping")
        return None, None
    try:
        df = pd.read_csv(p, sep="\t")
    except Exception as e:
        warn(f"muttime: failed to read: {e}")
        return None, None

    required = {"mutation_id", "ccf"}
    missing = required - set(df.columns)
    if missing:
        warn(f"muttime: missing columns {missing} — skipping")
        return None, None

    df["ccf"] = pd.to_numeric(df["ccf"], errors="coerce")
    df["tool"] = "muttime"
    # muttime doesn't cluster -- CLS (timing class) is the closest analogue,
    # kept as a label for display/crosswalk purposes only, not a real cluster
    df["cluster_label"] = df["CLS"] if "CLS" in df.columns else "unclassified"
    df["match_key"] = df["mutation_id"].apply(_match_key_from_id)

    n_ok = df["match_key"].notna().sum()
    info(f"muttime: {len(df)} mutations loaded, {n_ok} with usable mutation IDs")

    mut = df[["match_key","mutation_id","tool","cluster_label","ccf"]].dropna(subset=["ccf"])
    cluster_df = (mut.groupby("cluster_label")
                    .agg(n_mutations=("mutation_id","count"), ccf=("ccf","median"))
                    .reset_index())
    cluster_df["tool"] = "muttime"

    return mut, cluster_df[["tool","cluster_label","n_mutations","ccf"]]


TOOL_LOADERS = {
    "VIBER":       load_viber,
    "PyClone6":    load_pyclone6,
    "Orchard":     load_orchard,
    "PhylogicNDT": load_phylogic,
    "muttime":     load_muttime,
}


# ── page helpers ─────────────────────────────────────────────────────────────────

def _style(ax, title, xlabel, ylabel, fs=11):
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _header(fig, title, sample_id):
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=13, fontweight="bold", y=0.97)


def _caption(fig, text, y=0.02, fontsize=8.0):
    fig.text(0.04, y, text, ha="left", va="bottom", fontsize=fontsize,
             fontstyle="italic", color="#444444", wrap=True,
             transform=fig.transFigure)


def _save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def _no_data_page(pdf, title, sample_id, explanation):
    fig, ax = plt.subplots(figsize=(10, 6))
    _header(fig, title, sample_id)
    ax.axis("off")
    ax.text(0.5, 0.55, "Not enough data available", ha="center", va="center",
            fontsize=15, color="#AAAAAA", fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.40, explanation, ha="center", va="top", fontsize=9.5,
            color="#555555", wrap=True, transform=ax.transAxes)
    _save(pdf, fig)


# ── output TSVs ───────────────────────────────────────────────────────────────

def build_wide_comparison(mut_tables, clonal_threshold):
    """
    mut_tables: dict tool -> mut_df (with match_key, mutation_id, ccf).
    Returns a wide DataFrame: one row per match_key, columns
    {tool}_mutation_id, {tool}_cluster, {tool}_ccf, {tool}_is_clonal.
    """
    valid = {t: df for t, df in mut_tables.items() if df is not None and not df.empty
             and df["match_key"].notna().any()}
    if not valid:
        return pd.DataFrame()

    wide = None
    for tool in TOOL_ORDER:
        if tool not in valid:
            continue
        df = valid[tool].dropna(subset=["match_key"]).drop_duplicates("match_key")
        sub = df[["match_key","mutation_id","cluster_label","ccf"]].rename(columns={
            "mutation_id": f"{tool}_mutation_id",
            "cluster_label": f"{tool}_cluster",
            "ccf": f"{tool}_ccf",
        })
        sub[f"{tool}_is_clonal"] = sub[f"{tool}_ccf"] >= clonal_threshold
        wide = sub if wide is None else wide.merge(sub, on="match_key", how="outer")

    return wide if wide is not None else pd.DataFrame()


def write_outputs(wide, cluster_tables, purity_df, prefix):
    prefix_path = Path(prefix)
    prefix_path.parent.mkdir(parents=True, exist_ok=True)

    mut_path = f"{prefix}_per_mutation_comparison.tsv"
    if not wide.empty:
        wide.to_csv(mut_path, sep="\t", index=False)
        info(f"Written: {mut_path}  ({len(wide)} rows)")
    else:
        pd.DataFrame(columns=["match_key"]).to_csv(mut_path, sep="\t", index=False)
        warn(f"No per-mutation comparison data available — wrote empty file: {mut_path}")

    cl_path = f"{prefix}_per_cluster_summary.tsv"
    valid_cl = [df for df in cluster_tables.values() if df is not None and not df.empty]
    if valid_cl:
        pd.concat(valid_cl, ignore_index=True).to_csv(cl_path, sep="\t", index=False)
        info(f"Written: {cl_path}")
    else:
        pd.DataFrame(columns=["tool","cluster_label","n_mutations","ccf"]).to_csv(cl_path, sep="\t", index=False)
        warn(f"No cluster summary data available — wrote empty file: {cl_path}")

    pur_path = f"{prefix}_upstream_purity.tsv"
    if purity_df is not None and not purity_df.empty:
        purity_df.to_csv(pur_path, sep="\t", index=False)
        info(f"Written: {pur_path}")
    else:
        pd.DataFrame(columns=["source","purity"]).to_csv(pur_path, sep="\t", index=False)
        info(f"No upstream purity sources found — wrote empty file: {pur_path}")


# ── pages ──────────────────────────────────────────────────────────────────────

def page_overview(pdf, sample_id, mut_tables, cluster_tables, purity_df, clonal_threshold):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _header(fig, "Cross-Tool Comparison — Overview", sample_id)
    ax.axis("off")

    rows = [["Clonal-threshold used (uniform, all tools)", f"CCF ≥ {clonal_threshold:.2f}"]]
    for tool in TOOL_ORDER:
        mut = mut_tables.get(tool)
        cl  = cluster_tables.get(tool)
        if mut is None or mut.empty:
            rows.append([tool, "not available / did not run"])
            continue
        n_mut = len(mut)
        n_id  = mut["match_key"].notna().sum()
        n_cl  = cl["cluster_label"].nunique() if (cl is not None and not cl.empty) else "N/A"
        extra = f"{n_cl} clusters, " if TOOL_IS_CLUSTERING[tool] else "(timing classes, not clusters), "
        rows.append([tool, f"{n_mut} mutations ({n_id} matchable), {extra}"
                            f"clonal fraction ≥{clonal_threshold:.2f}: "
                            f"{100*mut['ccf'].ge(clonal_threshold).mean():.1f}%"])

    tbl = ax.table(cellText=rows, colLabels=["Tool", "Status"],
                   loc="upper center", cellLoc="left", bbox=[0.03, 0.55, 0.94, 0.38])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.0, 1.8)
    for j in range(2):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2):
            tbl[i, j].set_facecolor(shade)
    for i, tool in enumerate(["(threshold)"] + TOOL_ORDER, start=0):
        if i == 0:
            continue
        colour = TOOL_COLOURS.get(tool, "#CCCCCC")
        if mut_tables.get(tool) is not None and not mut_tables[tool].empty:
            tbl[i + 1, 0].set_facecolor(colour)
            tbl[i + 1, 0].set_text_props(color="white", fontweight="bold")

    n_running = sum(1 for t in TOOL_ORDER
                    if mut_tables.get(t) is not None and not mut_tables[t].empty)

    if purity_df is not None and not purity_df.empty:
        pur_rows = [[r["source"], f"{r['purity']:.4f}"] for _, r in purity_df.iterrows()]
        tbl2 = ax.table(cellText=pur_rows, colLabels=["Upstream purity source", "Value"],
                        loc="lower center", cellLoc="left", bbox=[0.03, 0.05, 0.94, 0.32])
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(9); tbl2.scale(1.0, 1.8)
        for j in range(2):
            tbl2[0, j].set_facecolor("#2C3E50")
            tbl2[0, j].set_text_props(color="white", fontweight="bold")
        spread = purity_df["purity"].max() - purity_df["purity"].min()
        note = (f"Purity spread across sources: {spread:.4f}. " +
                ("This is a candidate explanation if downstream tools disagree — "
                 "they may not all have used the same purity estimate."
                 if spread > 0.02 else
                 "All sources agree closely; differing purity input is unlikely "
                 "to explain any downstream disagreement."))
        _caption(fig, note, y=0.02)
    else:
        fig.text(0.5, 0.20, "No upstream purity files found (FACETS / CNAqc / "
                 "PhylogicNDT purity.txt / PyClone6 input) — purity cross-check skipped.",
                 ha="center", fontsize=9, color="#888888", transform=fig.transFigure)

    if n_running < 2:
        fig.text(0.5, 0.48, "Fewer than 2 tools produced usable output — "
                 "cross-tool comparison pages will be skipped.",
                 ha="center", fontsize=10, color="#C0392B", fontweight="bold",
                 transform=fig.transFigure)

    _save(pdf, fig)


def page_cluster_counts(pdf, sample_id, cluster_tables):
    tools = [t for t in TOOL_ORDER if TOOL_IS_CLUSTERING[t]
             and cluster_tables.get(t) is not None and not cluster_tables[t].empty]
    if not tools:
        _no_data_page(pdf, "Cluster Count Comparison", sample_id,
                      "No clustering tool (VIBER / PyClone6 / Orchard / PhylogicNDT) "
                      "produced usable cluster output.")
        return

    counts = [cluster_tables[t]["cluster_label"].nunique() for t in tools]
    colours = [TOOL_COLOURS[t] for t in tools]

    fig, ax = plt.subplots(figsize=(9, 6))
    _header(fig, "Number of Clusters Inferred per Tool", sample_id)
    bars = ax.bar(tools, counts, color=colours, edgecolor="white", alpha=0.88, width=0.55)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                str(c), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts) * 1.25)
    _style(ax, "Cluster Count per Tool  (muttime excluded — it classifies, not clusters)",
           "Tool", "Number of clusters")
    fig.subplots_adjust(bottom=0.18, top=0.88)
    _caption(fig,
        "Raw cluster count per tool. A large discrepancy (e.g. one tool finding 8 clusters, "
        "another finding 3) is common and can reflect genuinely different clustering "
        "resolutions, priors, or sensitivity to low-depth mutations — not necessarily an error. "
        "Check the ranked-CCF page next to see whether the overall clonal structure still "
        "looks similar despite the different counts.")
    _save(pdf, fig)


def page_clonal_fraction(pdf, sample_id, mut_tables, clonal_threshold):
    tools = [t for t in TOOL_ORDER if mut_tables.get(t) is not None and not mut_tables[t].empty]
    if not tools:
        _no_data_page(pdf, "Clonal Fraction per Tool", sample_id, "No tool produced usable CCF data.")
        return

    fracs = [100 * mut_tables[t]["ccf"].ge(clonal_threshold).mean() for t in tools]
    colours = [TOOL_COLOURS[t] for t in tools]

    fig, ax = plt.subplots(figsize=(9, 6))
    _header(fig, f"Fraction of Mutations Called Clonal (CCF ≥ {clonal_threshold:.2f})", sample_id)
    bars = ax.bar(tools, fracs, color=colours, edgecolor="white", alpha=0.88, width=0.55)
    for bar, f in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f"{f:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    _style(ax, f"% Mutations with CCF ≥ {clonal_threshold:.2f}, per tool",
           "Tool", "% mutations called clonal")
    fig.subplots_adjust(bottom=0.18, top=0.88)
    _caption(fig,
        f"Same clonal threshold (CCF ≥ {clonal_threshold:.2f}) applied uniformly to every "
        "tool's CCF estimate, so this comparison isn't affected by each tool's own internal "
        "threshold conventions. A tool that is a clear outlier here (much higher or lower "
        "clonal fraction than the rest) is calling this sample's overall clonal architecture "
        "quite differently from its peers.")
    _save(pdf, fig)


def page_ranked_cluster_ccf(pdf, sample_id, cluster_tables):
    tools = [t for t in TOOL_ORDER if TOOL_IS_CLUSTERING[t]
             and cluster_tables.get(t) is not None and not cluster_tables[t].empty]
    if not tools:
        _no_data_page(pdf, "Ranked Cluster CCF — Evolutionary Ordering", sample_id,
                      "No clustering tool produced usable cluster-level CCF output.")
        return

    ncols = min(len(tools), 4)
    nrows = int(np.ceil(len(tools) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.2*nrows+1), squeeze=False)
    _header(fig, "Clusters Ranked by CCF  (evolutionary ordering, highest first)", sample_id)
    fig.subplots_adjust(hspace=0.5, wspace=0.4, top=0.88, bottom=0.14)

    for idx, tool in enumerate(tools):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        cl = cluster_tables[tool].dropna(subset=["ccf"]).sort_values("ccf", ascending=False)
        y = np.arange(len(cl))
        ax.barh(y, cl["ccf"], color=TOOL_COLOURS[tool], alpha=0.85, edgecolor="white")
        ax.set_yticks(y)
        ax.set_yticklabels(cl["cluster_label"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.axvline(1.0, color="#E74C3C", ls="--", lw=1.0, alpha=0.6)
        _style(ax, tool, "CCF", "", fs=10)

    for idx in range(len(tools), nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "Each panel shows one tool's clusters ranked from most-clonal (top) to most-subclonal "
        "(bottom). This is a visual proxy for the inferred evolutionary ordering. Similar-looking "
        "'staircases' across panels (a few clusters near CCF=1, then a spread of lower values) "
        "suggest broad agreement on the tumour's evolutionary structure, even if exact cluster "
        "counts or CCF values differ. A tool whose staircase looks qualitatively different "
        "(e.g. one dominant cluster vs. many small ones) may be telling a different story "
        "about this tumour's clonal history.")
    _save(pdf, fig)


def page_concordance_matrix(pdf, sample_id, wide, tools_present, clonal_threshold):
    if len(tools_present) < 2:
        _no_data_page(pdf, "Pairwise Classification Concordance", sample_id,
                      "Fewer than 2 tools have matchable per-mutation data.")
        return

    n = len(tools_present)
    mat = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)
    for i, t1 in enumerate(tools_present):
        for j, t2 in enumerate(tools_present):
            if i == j:
                mat[i, j] = 1.0
                counts[i, j] = wide[f"{t1}_is_clonal"].notna().sum()
                continue
            c1 = f"{t1}_is_clonal"; c2 = f"{t2}_is_clonal"
            both = wide[[c1, c2]].dropna()
            if len(both) == 0:
                continue
            agree = (both[c1] == both[c2]).mean()
            mat[i, j] = agree
            counts[i, j] = len(both)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    _header(fig, "Pairwise Clonal/Subclonal Classification Agreement", sample_id)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(tools_present, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(tools_present, fontsize=9)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                txt = f"{mat[i,j]*100:.0f}%\n(n={counts[i,j]})"
                colour = "white" if mat[i,j] < 0.4 or mat[i,j] > 0.85 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color=colour)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Agreement fraction")
    fig.subplots_adjust(bottom=0.15, top=0.88, left=0.20)
    _caption(fig,
        "For each tool pair, the fraction of commonly-covered mutations where both tools "
        "agree on clonal vs subclonal (using the uniform threshold above). n = number of "
        "mutations both tools reported a CCF for. Low n means the comparison rests on few "
        "mutations and should be read cautiously.")
    _save(pdf, fig)


def page_pairwise_scatter(pdf, sample_id, wide, tools_present):
    pairs = [(t1, t2) for i, t1 in enumerate(tools_present)
             for t2 in tools_present[i+1:]]
    plotted = []
    for t1, t2 in pairs:
        c1, c2 = f"{t1}_ccf", f"{t2}_ccf"
        if c1 not in wide.columns or c2 not in wide.columns:
            continue
        both = wide[[c1, c2]].dropna()
        if len(both) >= 10:
            plotted.append((t1, t2, both))

    if not plotted:
        _no_data_page(pdf, "Pairwise CCF Correlation", sample_id,
                      "No tool pair has at least 10 commonly-covered mutations with CCF "
                      "values to correlate.")
        return

    ncols = min(len(plotted), 3)
    nrows = int(np.ceil(len(plotted) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6*ncols, 4.6*nrows+1), squeeze=False)
    _header(fig, "Pairwise CCF Correlation between Tools", sample_id)
    fig.subplots_adjust(hspace=0.55, wspace=0.4, top=0.90, bottom=0.14)

    for idx, (t1, t2, both) in enumerate(plotted):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        ax.scatter(both[f"{t1}_ccf"], both[f"{t2}_ccf"], s=14, alpha=0.5,
                   color="#4C72B0", edgecolors="none")
        ax.plot([0,1],[0,1], color="#AAAAAA", ls="--", lw=1.2)
        if len(both) >= 3:
            r = np.corrcoef(both[f"{t1}_ccf"], both[f"{t2}_ccf"])[0,1]
            ax.text(0.05, 0.93, f"r = {r:.2f}\nn = {len(both)}",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax.set_xlim(0,1.05); ax.set_ylim(0,1.05)
        _style(ax, f"{t1} vs {t2}", f"{t1} CCF", f"{t2} CCF", fs=9.5)

    for idx in range(len(plotted), nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "Each point = one mutation covered by both tools. Points on the dashed y=x line "
        "indicate perfect agreement. A cloud shifted consistently above or below the line "
        "(rather than scattered around it) suggests a systematic calibration difference "
        "between the two tools — e.g. a purity/copy-number correction handled differently — "
        "rather than mutation-by-mutation noise. r = Pearson correlation (not a composite "
        "score, just the standard linear correlation coefficient).")
    _save(pdf, fig)


def page_ccf_trajectories(pdf, sample_id, wide, tools_present, max_mutations=250):
    ccf_cols = [f"{t}_ccf" for t in tools_present if f"{t}_ccf" in wide.columns]
    if len(ccf_cols) < 3:
        _no_data_page(pdf, "Per-mutation CCF Trajectories across Tools", sample_id,
                      "Fewer than 3 tools have CCF data — a trajectory comparison needs "
                      "at least 3 tools to be informative.")
        return

    sub = wide.dropna(subset=ccf_cols, thresh=3)[ccf_cols + ["match_key"]].copy()
    sub = sub.dropna(subset=ccf_cols)  # require ALL selected tools for a clean trajectory view
    if len(sub) == 0:
        # relax: allow partial coverage across the >=3 tools
        sub = wide.dropna(subset=ccf_cols, thresh=3)[ccf_cols + ["match_key"]].copy()
    if len(sub) == 0:
        _no_data_page(pdf, "Per-mutation CCF Trajectories across Tools", sample_id,
                      "No mutations are covered by at least 3 tools simultaneously.")
        return

    if len(sub) > max_mutations:
        sub = sub.sample(max_mutations, random_state=42)
        subtitle_note = f" (random sample of {max_mutations} shown)"
    else:
        subtitle_note = ""

    tools_used = [t for t in tools_present if f"{t}_ccf" in ccf_cols]
    x = np.arange(len(tools_used))

    fig, ax = plt.subplots(figsize=(max(9, 2*len(tools_used)), 7))
    _header(fig, f"Per-mutation CCF across Tools{subtitle_note}", sample_id)

    for _, row in sub.iterrows():
        vals = [row[f"{t}_ccf"] for t in tools_used]
        ax.plot(x, vals, color="#888888", alpha=0.12, lw=0.8, zorder=1)

    for i, t in enumerate(tools_used):
        vals = sub[f"{t}_ccf"].dropna()
        ax.scatter([i]*len(vals), vals, color=TOOL_COLOURS.get(t,"#4C72B0"),
                   s=10, alpha=0.6, zorder=3, edgecolors="none")
        ax.scatter(i, vals.median(), color="black", marker="D", s=60, zorder=4)

    ax.set_xticks(x); ax.set_xticklabels(tools_used, fontsize=10)
    ax.set_ylim(0, 1.05)
    _style(ax, "CCF Trajectory per Mutation across Tools\n(grey lines connect the same mutation; black diamond = median per tool)",
           "Tool", "CCF")
    fig.subplots_adjust(bottom=0.18, top=0.85)
    _caption(fig,
        "Each grey line follows one mutation's CCF as reported by each tool. Mostly-flat, "
        "parallel lines mean tools agree mutation-by-mutation. If one tool's column is "
        "visibly and consistently shifted up or down relative to the others (rather than "
        "individual lines crossing randomly), that tool is telling a systematically "
        "different story for this sample — worth checking that tool's purity/copy-number "
        "inputs and its own QC report before trusting it over the others.")
    _save(pdf, fig)


def page_muttime_crosswalk(pdf, sample_id, wide, tools_present):
    if "muttime_cluster" not in wide.columns:
        _no_data_page(pdf, "Mutation Timing (muttime) vs Clustering Tools", sample_id,
                      "muttime did not run, or produced no usable output.")
        return

    other_tools = [t for t in tools_present if t != "muttime" and f"{t}_is_clonal" in wide.columns]
    if not other_tools:
        _no_data_page(pdf, "Mutation Timing (muttime) vs Clustering Tools", sample_id,
                      "No clustering tool has matchable per-mutation clonal/subclonal calls "
                      "to cross-tabulate against muttime's timing classes.")
        return

    cls_order = [c for c in ["early clonal","late clonal","clonal [NA]","subclonal","unclassified"]
                 if c in wide["muttime_cluster"].dropna().unique()]
    if not cls_order:
        cls_order = sorted(wide["muttime_cluster"].dropna().unique())

    ncols = min(len(other_tools), 3)
    nrows = int(np.ceil(len(other_tools) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8*ncols, 4.5*nrows+1), squeeze=False)
    _header(fig, "muttime Timing Classes vs Other Tools' Clonal Calls", sample_id)
    fig.subplots_adjust(hspace=0.6, wspace=0.4, top=0.88, bottom=0.16)

    for idx, tool in enumerate(other_tools):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = wide.dropna(subset=["muttime_cluster", f"{tool}_is_clonal"])
        if sub.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, "No overlapping mutations", ha="center", transform=ax.transAxes)
            continue
        clonal_counts    = [sub[(sub["muttime_cluster"]==c) & (sub[f"{tool}_is_clonal"]==True)].shape[0]
                            for c in cls_order]
        subclonal_counts = [sub[(sub["muttime_cluster"]==c) & (sub[f"{tool}_is_clonal"]==False)].shape[0]
                            for c in cls_order]
        x = np.arange(len(cls_order))
        ax.bar(x, clonal_counts, color=TOOL_COLOURS.get(tool,"#4C72B0"), alpha=0.85,
               label=f"{tool}: clonal", edgecolor="white")
        ax.bar(x, subclonal_counts, bottom=clonal_counts, color=TOOL_COLOURS.get(tool,"#4C72B0"),
               alpha=0.35, label=f"{tool}: subclonal", edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(cls_order, rotation=30, ha="right", fontsize=7.5)
        ax.legend(fontsize=7)
        _style(ax, tool, "muttime timing class", "Mutation count", fs=10)

    for idx in range(len(other_tools), nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "For each muttime timing class, how many of those mutations each clustering tool "
        "independently calls clonal (solid) vs subclonal (faded). Good agreement looks like: "
        "'early/late clonal' and 'clonal [NA]' mutations are mostly solid; 'subclonal' mutations "
        "are mostly faded. muttime's timing classes are derived from copy-number + CCF thresholds "
        "(see muttime's own report), not from these clustering tools, so this is an independent "
        "cross-check of the evolutionary timeline.")
    _save(pdf, fig)


def page_top_disagreements(pdf, sample_id, wide, tools_present, top_n=20):
    ccf_cols = [f"{t}_ccf" for t in tools_present if f"{t}_ccf" in wide.columns]
    if len(ccf_cols) < 2:
        _no_data_page(pdf, "Largest Per-mutation Disagreements", sample_id,
                      "Fewer than 2 tools have CCF data to compare.")
        return

    sub = wide.dropna(subset=ccf_cols, thresh=2).copy()
    if sub.empty:
        _no_data_page(pdf, "Largest Per-mutation Disagreements", sample_id,
                      "No mutations are covered by at least 2 tools simultaneously.")
        return

    sub["ccf_range"] = sub[ccf_cols].max(axis=1) - sub[ccf_cols].min(axis=1)
    sub = sub.sort_values("ccf_range", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11, max(6, 0.35*len(sub)+2)))
    _header(fig, f"Top {len(sub)} Mutations with Largest CCF Disagreement across Tools", sample_id)
    ax.axis("off")

    tools_used = [t for t in tools_present if f"{t}_ccf" in ccf_cols]
    col_labels = ["mutation_id"] + tools_used + ["range"]
    rows = []
    for _, r in sub.iterrows():
        mid = next((r[f"{t}_mutation_id"] for t in tools_used
                    if pd.notna(r.get(f"{t}_mutation_id"))), r["match_key"])
        row = [str(mid)]
        for t in tools_used:
            v = r.get(f"{t}_ccf")
            row.append(f"{v:.3f}" if pd.notna(v) else "—")
        row.append(f"{r['ccf_range']:.3f}")
        rows.append(row)

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(shade)

    fig.subplots_adjust(top=0.90, bottom=0.05)
    _caption(fig,
        "Mutations ranked by the spread between the highest and lowest CCF reported across "
        "tools (only mutations covered by ≥2 tools shown). These are the most concrete, "
        "checkable cases if you want to dig into why tools disagree for this sample — "
        "look up these specific mutations in each tool's own detailed report.", y=0.01)
    _save(pdf, fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-tool comparison of clonal reconstruction output "
                     "(VIBER, PyClone6, Orchard, PhylogicNDT, muttime)."
    )
    parser.add_argument("--sample-dir", "-d", default=None,
                        help="Directory containing per-tool subfolders "
                             "(viber/, pyclone6/, orchard/, phylogic/, muttime/, "
                             "facets/, cnaqc/)")
    parser.add_argument("--viber-dir",    default=None)
    parser.add_argument("--pyclone6-dir", default=None)
    parser.add_argument("--orchard-dir",  default=None)
    parser.add_argument("--phylogic-dir", default=None)
    parser.add_argument("--muttime-dir",  default=None)
    parser.add_argument("--facets-dir",   default=None)
    parser.add_argument("--cnaqc-dir",    default=None)
    parser.add_argument("--sample", "-s", required=True)
    parser.add_argument("--clonal-threshold", type=float, default=0.70,
                        help="Uniform CCF threshold for clonal classification, "
                             "applied identically to every tool (default: 0.70)")
    parser.add_argument("--output-prefix", "-o", default=None,
                        help="Output path prefix (default: <sample>_compare/<sample>)")
    args = parser.parse_args()

    def _resolve_dir(explicit, subname):
        if explicit:
            return explicit
        if args.sample_dir:
            candidate = Path(args.sample_dir) / subname
            return str(candidate) if candidate.exists() else None
        return None

    dirs = {
        "VIBER":       _resolve_dir(args.viber_dir,    "viber"),
        "PyClone6":    _resolve_dir(args.pyclone6_dir, "pyclone6"),
        "Orchard":     _resolve_dir(args.orchard_dir,  "orchard"),
        "PhylogicNDT": _resolve_dir(args.phylogic_dir, "phylogic"),
        "muttime":     _resolve_dir(args.muttime_dir,  "muttime"),
    }
    facets_dir = _resolve_dir(args.facets_dir, "facets")
    cnaqc_dir  = _resolve_dir(args.cnaqc_dir,  "cnaqc")

    sample = args.sample
    prefix = args.output_prefix or f"{sample}_compare/{sample}"

    print(f"[compare] Sample: {sample}")
    print(f"[compare] Clonal threshold (uniform): CCF ≥ {args.clonal_threshold}")

    mut_tables = {}
    cluster_tables = {}
    for tool in TOOL_ORDER:
        d = dirs[tool]
        if d is None:
            info(f"{tool}: directory not found/specified — skipping")
            mut_tables[tool] = None
            cluster_tables[tool] = None
            continue
        mut, cl = TOOL_LOADERS[tool](d, sample)
        mut_tables[tool] = mut
        cluster_tables[tool] = cl

    n_running = sum(1 for t in TOOL_ORDER
                    if mut_tables.get(t) is not None and not mut_tables[t].empty)
    print(f"[compare] {n_running}/5 tools produced usable output: "
          f"{[t for t in TOOL_ORDER if mut_tables.get(t) is not None and not mut_tables[t].empty]}")

    purity_df = load_upstream_purity(sample, facets_dir, cnaqc_dir, dirs["PhylogicNDT"], dirs["PyClone6"])

    wide = build_wide_comparison(mut_tables, args.clonal_threshold)
    tools_present = [t for t in TOOL_ORDER if f"{t}_ccf" in wide.columns] if not wide.empty else []

    print(f"[compare] Writing output TSVs to: {prefix}_*")
    write_outputs(wide, cluster_tables, purity_df, prefix)

    pdf_path = f"{prefix}_comparison_report.pdf"
    print(f"[compare] Writing report: {pdf_path}")

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.5,
    })

    with PdfPages(pdf_path) as pdf:
        info_dict = pdf.infodict()
        info_dict["Title"]  = f"Clonal Tool Comparison — {sample}"
        info_dict["Author"] = "compare_clonal_tools.py"

        page_overview(pdf, sample, mut_tables, cluster_tables, purity_df, args.clonal_threshold)
        page_cluster_counts(pdf, sample, cluster_tables)
        page_clonal_fraction(pdf, sample, mut_tables, args.clonal_threshold)
        page_ranked_cluster_ccf(pdf, sample, cluster_tables)

        if not wide.empty and len(tools_present) >= 2:
            page_concordance_matrix(pdf, sample, wide, tools_present, args.clonal_threshold)
            page_pairwise_scatter(pdf, sample, wide, tools_present)
            page_ccf_trajectories(pdf, sample, wide, tools_present)
            page_muttime_crosswalk(pdf, sample, wide, tools_present)
            page_top_disagreements(pdf, sample, wide, tools_present)
        else:
            _no_data_page(pdf, "Cross-Tool Mutation Comparison", sample,
                          "Fewer than 2 tools have matchable per-mutation CCF data — "
                          "detailed cross-tool comparison pages are skipped. "
                          "See the overview page for per-tool status.")

    print(f"[compare] Done — {pdf_path}")


if __name__ == "__main__":
    main()