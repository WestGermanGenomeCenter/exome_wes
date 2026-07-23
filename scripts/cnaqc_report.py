#!/usr/bin/env python3
"""
cnaqc_report.py — Human-understandable PDF report explaining CNAqc output.

CNAqc validates whether allele-specific copy number segments and tumour
purity are consistent with the observed VAF distribution of somatic
mutations, and estimates per-mutation Cancer Cell Fractions (CCFs) with
their uncertainty. This script reads the files produced by
extract_cnaqc_rds.R and builds a PDF that walks through what CNAqc
computed and, where possible, illustrates *why* -- i.e. what VAF value
each copy-number state predicts, and how the observed data compares.

Usage:
    python cnaqc_report.py \
        --dir     sample_cnaqc_export/  \
        [--qc-txt sample_cnaqc_qc.txt]  \
        [--sample sample1]              \
        [--output sample1_cnaqc_report.pdf]

    # or point to files individually:
    python cnaqc_report.py \
        --mutations mutations.tsv \
        --cna       cna_clonal.tsv \
        --karyotype-summary karyotype_summary.tsv \
        --metadata  metadata.json \
        [--peaks-matches peaks_matches.tsv] \
        [--qc-txt   sample_cnaqc_qc.txt]

Expected files (written by extract_cnaqc_rds.R):
    mutations.tsv          : chr from to ref alt NV DP VAF type karyotype segment_id [CCF]
    cna_clonal.tsv          : chr from to Major minor CCF length segment_id n subclonal
    cna_subclonal.tsv       : same columns, subclonal segments (optional)
    karyotype_summary.tsv   : karyotype n_segments total_bp n_mutations
    peaks_matches.tsv       : CNAqc::analyze_peaks() results (optional)
    metadata.json           : sample, purity, ploidy, n_mutations, ...
    <sample>_cnaqc_qc.txt   : plain-text automated QC verdict (optional, tab-separated)

Conda deps (all conda-forge):
    conda install -c conda-forge pandas numpy matplotlib scipy
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

# constants
CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHR_LENGTHS = {
    "chr1":249e6,"chr2":242e6,"chr3":198e6,"chr4":190e6,"chr5":182e6,
    "chr6":171e6,"chr7":159e6,"chr8":145e6,"chr9":138e6,"chr10":134e6,
    "chr11":135e6,"chr12":133e6,"chr13":115e6,"chr14":107e6,"chr15":102e6,
    "chr16":90e6,"chr17":83e6,"chr18":80e6,"chr19":59e6,"chr20":63e6,
    "chr21":48e6,"chr22":51e6,"chrX":156e6,"chrY":58e6,
}

KAR_PALETTE = [
    "#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860",
    "#DA8BC3","#64B5CD","#CCB974","#E377C2","#7F7F7F","#BCBD22",
]

def warn(msg):  print(f"  [WARN] {msg}", file=sys.stderr)
def info(msg):  print(f"  [INFO] {msg}", file=sys.stderr)


def _kar_palette(karyotypes):
    labels = sorted(karyotypes)
    return {k: KAR_PALETTE[i % len(KAR_PALETTE)] for i, k in enumerate(labels)}


def _normalise_chrom(s):
    s = str(s).strip()
    return s if s.startswith("chr") else f"chr{s}"


def _parse_karyotype(kar):
    """'3:2' -> (3, 2). Returns (nan, nan) if malformed."""
    try:
        parts = str(kar).split(":")
        return int(parts[0]), int(parts[1])
    except Exception:
        return np.nan, np.nan


def _expected_vaf(purity, major, minor, m):
    """
    Theoretical VAF for a mutation present on m allele copies, in a segment
    with given Major/minor copy number, at the given tumour purity.

        VAF = (m * purity) / (2*(1-purity) + purity*(Major+minor))

    This is the same combinatorial relationship CNAqc uses internally
    (Dentro et al. 2017 / Gerstung et al. 2020 formulation).
    """
    total_cn = major + minor
    denom = 2 * (1 - purity) + purity * total_cn
    if denom <= 0:
        return np.nan
    return (m * purity) / denom


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
    """A gracefully-degraded page shown when required data is missing."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _header(fig, title, sample_id)
    ax.axis("off")
    ax.text(0.5, 0.55, "Data not available", ha="center", va="center",
            fontsize=16, color="#AAAAAA", fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.5, 0.40, explanation, ha="center", va="top", fontsize=9.5,
            color="#555555", wrap=True, transform=ax.transAxes)
    _save(pdf, fig)

# I/O

def _safe_read_tsv(path, label):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        info(f"{label} not found: {p}")
        return None
    try:
        df = pd.read_csv(p, sep="\t")
        if df.empty:
            warn(f"{label} is empty: {p}")
            return None
        info(f"Loaded {label}: {len(df)} rows")
        return df
    except Exception as e:
        warn(f"Could not read {label} ({p}): {e}")
        return None


def load_metadata(path):
    if path is None or not Path(path).exists():
        warn("metadata.json not found -- sample-level context will be limited")
        return {}
    try:
        with open(path) as f:
            meta = json.load(f)
        info(f"Loaded metadata.json: {list(meta.keys())}")
        return meta
    except Exception as e:
        warn(f"Could not parse metadata.json: {e}")
        return {}


def load_qc_txt(path):
    if path is None or not Path(path).exists():
        info("qc.txt not supplied -- QC verdict page will be skipped")
        return None
    try:
        df = pd.read_csv(path, sep="\t")
        if df.empty:
            warn("qc.txt is empty")
            return None
        info(f"Loaded qc.txt: {list(df.columns)}")
        return df.iloc[0].to_dict()
    except Exception as e:
        warn(f"Could not parse qc.txt: {e}")
        return None


def load_mutations(path):
    df = _safe_read_tsv(path, "mutations.tsv")
    if df is None:
        return None

    required = {"VAF", "DP", "NV", "karyotype"}
    missing = required - set(df.columns)
    if missing:
        warn(f"mutations.tsv missing expected columns: {missing}")

    if "chr" in df.columns:
        df["chr"] = df["chr"].apply(_normalise_chrom)
    if "from" in df.columns:
        df = df.rename(columns={"from": "pos"})
    elif "pos" not in df.columns and "to" in df.columns:
        df["pos"] = df["to"]

    for col in ("VAF", "DP", "NV"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # sanity checks
    n0 = len(df)
    if "VAF" in df.columns:
        bad_vaf = (df["VAF"] < 0) | (df["VAF"] > 1) | df["VAF"].isna()
        if bad_vaf.any():
            warn(f"{bad_vaf.sum()}/{n0} mutations have VAF outside [0,1] or missing")
    if {"NV", "DP"}.issubset(df.columns):
        bad_depth = df["NV"] > df["DP"]
        if bad_depth.any():
            warn(f"{bad_depth.sum()}/{n0} mutations have NV (alt reads) > DP (total depth) -- "
                 f"check upstream VCF parsing")
    if "karyotype" in df.columns:
        parsed = df["karyotype"].apply(_parse_karyotype)
        n_bad_kar = parsed.apply(lambda t: np.isnan(t[0])).sum()
        if n_bad_kar > 0:
            warn(f"{n_bad_kar}/{n0} mutations have unparsable karyotype strings")
        df["major_cn"] = parsed.apply(lambda t: t[0])
        df["minor_cn"] = parsed.apply(lambda t: t[1])
        df["total_cn"] = df["major_cn"] + df["minor_cn"]

    return df


def load_cna(clonal_path, subclonal_path):
    clonal = _safe_read_tsv(clonal_path, "cna_clonal.tsv")
    subcl  = _safe_read_tsv(subclonal_path, "cna_subclonal.tsv")

    parts = [d for d in (clonal, subcl) if d is not None]
    if not parts:
        return None
    cna = pd.concat(parts, ignore_index=True)

    if "chr" in cna.columns:
        cna["chr"] = cna["chr"].apply(_normalise_chrom)
    if "from" in cna.columns:
        cna = cna.rename(columns={"from": "start"})
    if "to" in cna.columns:
        cna = cna.rename(columns={"to": "end"})
    for col in ("Major", "minor", "CCF", "length"):
        if col in cna.columns:
            cna[col] = pd.to_numeric(cna[col], errors="coerce")

    if {"Major", "minor"}.issubset(cna.columns):
        bad = cna["minor"] > cna["Major"]
        if bad.any():
            warn(f"{bad.sum()} CNA segments have minor > Major -- check R extractor / CNAqc object")

    if "subclonal" not in cna.columns:
        cna["subclonal"] = False

    return cna

# pages

def page_overview(pdf, meta, qc, mut, cna, kar_summary, sample_id):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _header(fig, "CNAqc Report - Overview", sample_id)
    ax.axis("off")

    purity = meta.get("purity", np.nan)
    ploidy = meta.get("ploidy", np.nan)
    n_mut  = meta.get("n_mutations", len(mut) if mut is not None else "N/A")
    n_cna  = meta.get("n_cna", len(cna) if cna is not None else "N/A")
    has_sub = meta.get("has_subclonal_CNA", None)

    def _fmt(v, fmt):
        try:
            if v is None: return "N/A"
            fv = float(v)
            if np.isnan(fv): return "N/A"
            return fmt.format(fv)
        except Exception:
            return "N/A"

    rows = [
        ["Sample",                  str(meta.get("sample", sample_id))],
        ["Reference genome",        str(meta.get("reference_genome", "N/A"))],
        ["Tumour purity",           _fmt(purity, "{:.4f}")],
        ["Tumour ploidy",           _fmt(ploidy, "{:.2f}")],
        ["Total mutations",         str(n_mut)],
        ["Total CNA segments",      str(n_cna)],
        ["Has subclonal CNA?",      str(has_sub) if has_sub is not None else "N/A"],
        ["Most prevalent karyotype (by genome bp)",
                                     str(meta.get("most_prevalent_karyotype", "N/A"))],
        ["Karyotype with most mutations",
                                     str(meta.get("most_mutations_karyotype", "N/A"))],
        ["Peak analysis available?",str(meta.get("has_peaks_analysis", False))],
        ["Per-mutation CCF available?", str(meta.get("has_ccf", False))],
    ]

    tbl = ax.table(cellText=rows, colLabels=["Property","Value"],
                   loc="upper center", cellLoc="left", bbox=[0.05,0.42,0.9,0.50])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1.0, 1.7)
    for j in range(2):
        tbl[0,j].set_facecolor("#2C3E50")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2): tbl[i,j].set_facecolor(shade)

    explanation = (
        "WHAT CNAqc DOES\n"
        "CNAqc checks whether the allele-specific copy number (CNA) segments and the "
        "tumour purity estimate are internally consistent with the VAF distribution of "
        "somatic mutations. The core idea: on a copy-neutral diploid segment (karyotype "
        "1:1), a heterozygous clonal mutation is expected at VAF is approximately purity/2. "
        "On amplified or LOH segments the expected VAF shifts in a predictable way "
        "depending on how many allele copies carry the mutation (its 'multiplicity'). "
        "If the observed VAF histogram doesn't peak where the copy number model "
        "predicts, that's evidence the purity, ploidy, or CN calls need revisiting.\n\n"
        "CNAqc also computes per-mutation Cancer Cell Fraction (CCF) -- the proportion "
        "of tumour cells carrying that mutation -- by inverting this same relationship, "
        "and flags mutations for which CCF cannot be confidently resolved from the data.\n\n"
        "The following pages walk through this sample's data: genome composition by "
        "copy-number state, the VAF patterns within each state, how they compare to "
        "theoretical expectations, and (if available) CNAqc's own peak-matching QC verdict."
    )
    fig.text(0.05, 0.36, explanation, ha="left", va="top", fontsize=9,
             color="#222222", wrap=True, transform=fig.transFigure,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#EBF5FB",
                       edgecolor="#AED6F1", alpha=0.85))
    _save(pdf, fig)


def page_qc_verdict(pdf, qc, sample_id):
    if qc is None:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    _header(fig, "Automated QC Verdict", sample_id)
    ax.axis("off")

    field_help = {
        "sample":       "Sample identifier.",
        "purity_used":  "Tumour purity value used for this QC run.",
        "n_mutations":  "Number of mutations considered.",
        "n_mapped":     "Number of mutations successfully mapped to a CNA segment. "
                        "NA usually means the mapping step was not run or failed silently.",
        "lambda_score": "A goodness-of-fit score for the peak-matching QC (higher is usually "
                        "better fit between observed and expected VAF peaks). NA means "
                        "analyze_peaks() was not run, or produced no usable peaks (e.g. too "
                        "few mutations per karyotype, or purity/CN too ambiguous).",
        "epsilon":      "Tolerance window (in VAF units) used when matching observed peaks "
                        "to theoretical expected peaks. Smaller = stricter matching.",
        "cnaqc_pass":   "Overall PASS/FAIL verdict for this sample's CN/purity/VAF "
                        "consistency. NA means the automated verdict could not be computed "
                        "(often because lambda_score is also NA).",
    }

    rows = []
    for k, v in qc.items():
        v_str = "NA" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        rows.append([k, v_str, field_help.get(k, "")])

    tbl = ax.table(cellText=rows, colLabels=["Field","Value","Meaning"],
                   loc="center", cellLoc="left", bbox=[0.02,0.15,0.96,0.75])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for j in range(3):
        tbl[0,j].set_facecolor("#2C3E50")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(3): tbl[i,j].set_facecolor(shade)

    n_na = sum(1 for v in qc.values() if v is None or (isinstance(v,float) and np.isnan(v)))
    if n_na > 0:
        note = (f"{n_na} field(s) are NA in this QC file. This typically means the peak-matching "
                f"step (CNAqc::analyze_peaks()) either was not run, or could not produce a "
                f"confident verdict -- see the 'Peak Analysis' page later in this report for what "
                f"data was available to attempt it.")
        _caption(fig, note, y=0.03)
    _save(pdf, fig)


def _safe_count_label(v):
    """Format a bar-chart count label, tolerating NaN/None without crashing."""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "?"
        return str(int(round(float(v))))
    except (TypeError, ValueError):
        return "?"


def page_karyotype_segments(pdf, kar_summary, cna, sample_id, pal):
    if kar_summary is None and cna is None:
        _no_data_page(pdf, "Genome Composition - Segment Counts", sample_id,
                      "Neither karyotype_summary.tsv nor cna_clonal.tsv was provided.")
        return

    if kar_summary is not None and "n_segments" in kar_summary.columns:
        ks = kar_summary.dropna(subset=["karyotype"]).copy()
        n_before = len(ks)
        ks = ks.dropna(subset=["n_segments"])
        n_dropped = n_before - len(ks)
        if n_dropped > 0:
            warn(f"karyotype_summary.tsv: {n_dropped} karyotype row(s) have no n_segments "
                 f"value (present in mutations but not matched to a CNA segment during the "
                 f"R-side merge) — excluded from this segment-count chart. "
                 f"They still appear on the 'Mutations per Karyotype' page.")
        ks = ks.sort_values("n_segments", ascending=False)
        if ks.empty:
            _no_data_page(pdf, "Genome Composition - Segment Counts", sample_id,
                          "karyotype_summary.tsv has no rows with a valid n_segments value "
                          "after dropping unmatched entries.")
            return
        karyotypes, counts = ks["karyotype"].tolist(), ks["n_segments"].tolist()
    elif cna is not None and {"Major","minor"}.issubset(cna.columns):
        vc = (cna["Major"].astype("Int64").astype(str) + ":" + cna["minor"].astype("Int64").astype(str)).value_counts()
        karyotypes, counts = vc.index.tolist(), vc.values.tolist()
    else:
        _no_data_page(pdf, "Genome Composition - Segment Counts", sample_id,
                      "No usable segment-count data found.")
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    _header(fig, "Genome Composition - CNA Segments per Karyotype", sample_id)
    colours = [pal.get(k, "#888888") for k in karyotypes]
    bars = ax.bar(karyotypes, counts, color=colours, edgecolor="white", alpha=0.87)
    max_count = max([c for c in counts if c is not None and not (isinstance(c,float) and np.isnan(c))], default=1)
    for bar, c in zip(bars, counts):
        height = 0 if (c is None or (isinstance(c,float) and np.isnan(c))) else bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, height+max_count*0.01,
                _safe_count_label(c), ha="center", fontsize=8)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    _style(ax, "Number of CNA Segments per Karyotype (Major:minor)",
           "Karyotype", "Segment count")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    _caption(fig,
        "Karyotype notation is Major:minor allele copy number. '1:1' is copy-neutral "
        "diploid (2 total copies, balanced). '2:0' is loss of one allele with the other "
        "amplified (net 2 copies but only 1 unique allele -- LOH). '2:1', '3:1' etc. are "
        "amplifications. This chart counts distinct CNA segments, not genome length -- "
        "see the next page for that.")
    _save(pdf, fig)


def page_karyotype_bp(pdf, kar_summary, sample_id, pal):
    if kar_summary is None or "total_bp" not in kar_summary.columns:
        _no_data_page(pdf, "Genome Composition - Base Pairs Covered", sample_id,
                      "karyotype_summary.tsv was not provided, or has no total_bp column.")
        return

    ks = kar_summary.dropna(subset=["karyotype","total_bp"]).sort_values("total_bp", ascending=False)
    if ks.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 6.5))
    _header(fig, "Genome Composition - Base Pairs Covered per Karyotype", sample_id)
    mb = ks["total_bp"] / 1e6
    colours = [pal.get(k, "#888888") for k in ks["karyotype"]]
    bars = ax.bar(ks["karyotype"], mb, color=colours, edgecolor="white", alpha=0.87)
    total_mb = mb.sum()
    for bar, v in zip(bars, mb):
        pct = 100*v/total_mb if total_mb>0 else 0
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(mb)*0.01,
                f"{pct:.0f}%", ha="center", fontsize=8)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    _style(ax, f"Genome Length per Karyotype  (total {total_mb:.0f} Mb covered)",
           "Karyotype", "Length (Mb)")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    _caption(fig,
        "This shows what fraction of the (segmented) genome sits in each copy-number "
        "state, weighted by base pairs rather than segment count -- a few large segments "
        "can dominate even if many small segments exist elsewhere. This is usually a more "
        "biologically meaningful view of how much of the tumour genome is diploid vs "
        "altered than the segment-count chart on the previous page.")
    _save(pdf, fig)


def page_mutations_per_karyotype(pdf, mut, sample_id, pal):
    if mut is None or "karyotype" not in mut.columns:
        _no_data_page(pdf, "Mutations per Karyotype", sample_id,
                      "mutations.tsv was not provided, or has no karyotype column.")
        return

    vc = mut["karyotype"].value_counts()
    fig, ax = plt.subplots(figsize=(11, 6.5))
    _header(fig, "Mutations per Karyotype", sample_id)
    colours = [pal.get(k, "#888888") for k in vc.index]
    bars = ax.bar(vc.index, vc.values, color=colours, edgecolor="white", alpha=0.87)
    max_count = max(vc.values) if len(vc.values) else 1
    for bar, c in zip(bars, vc.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max_count*0.01,
                _safe_count_label(c), ha="center", fontsize=8)
    ax.tick_params(axis="x", rotation=40, labelsize=9)
    _style(ax, "Somatic Mutations per Karyotype (Major:minor)",
           "Karyotype", "Mutation count")
    fig.subplots_adjust(bottom=0.20, top=0.88)
    _caption(fig,
        "Number of mutations CNAqc mapped to each copy-number state. CNAqc's peak-QC "
        "is performed independently within each karyotype bin, so bins with very few "
        "mutations (rule of thumb: below ~50-100) often can't produce a statistically "
        "confident peak-matching verdict, even if the underlying calls are fine.")
    _save(pdf, fig)


def page_vaf_by_karyotype(pdf, mut, sample_id, pal, top_n=8):
    if mut is None or "karyotype" not in mut.columns or "VAF" not in mut.columns:
        _no_data_page(pdf, "VAF Distribution per Karyotype", sample_id,
                      "mutations.tsv missing VAF or karyotype columns.")
        return

    top_kars = mut["karyotype"].value_counts().head(top_n).index.tolist()
    ncols = min(len(top_kars), 3) if top_kars else 1
    nrows = int(np.ceil(len(top_kars)/ncols)) if top_kars else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 3.8*nrows+1.2), squeeze=False)
    _header(fig, f"VAF Distribution per Karyotype  (top {len(top_kars)} by mutation count)", sample_id)
    fig.subplots_adjust(hspace=0.55, wspace=0.35, top=0.90, bottom=0.14)

    bins = np.linspace(0, 1, 41)
    for idx, kar in enumerate(top_kars):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = mut[mut["karyotype"]==kar]["VAF"].dropna()
        colour = pal.get(kar, "#888888")
        ax.hist(sub, bins=bins, color=colour, alpha=0.78, edgecolor="white")
        if len(sub) > 0:
            ax.axvline(sub.median(), color="black", lw=1.5, ls="--",
                       label=f"median={sub.median():.2f}")
            ax.legend(fontsize=7)
        _style(ax, f"{kar}  (n={len(sub)})", "VAF", "Count", fs=9.5)
    for idx in range(len(top_kars), nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "Raw VAF distribution for mutations in each karyotype bin. On the next page these "
        "are compared against the theoretical VAF values predicted from purity + copy "
        "number, which is the core logic CNAqc uses for QC.")
    _save(pdf, fig)


def page_expected_vs_observed(pdf, mut, meta, sample_id, pal, top_n=8):
    if mut is None or "karyotype" not in mut.columns or "VAF" not in mut.columns:
        _no_data_page(pdf, "Expected vs Observed VAF Peaks", sample_id,
                      "mutations.tsv missing VAF or karyotype columns.")
        return

    purity = meta.get("purity", np.nan)
    try:
        purity_f = float(purity)
    except Exception:
        purity_f = np.nan
    if np.isnan(purity_f):
        _no_data_page(pdf, "Expected vs Observed VAF Peaks", sample_id,
                      "Tumour purity not available in metadata.json -- cannot compute "
                      "theoretical expected VAF peaks without it.")
        return

    top_kars = mut["karyotype"].value_counts().head(top_n).index.tolist()
    ncols = min(len(top_kars), 3) if top_kars else 1
    nrows = int(np.ceil(len(top_kars)/ncols)) if top_kars else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 3.8*nrows+1.4), squeeze=False)
    _header(fig, f"Expected vs Observed VAF Peaks  (purity={purity_f:.3f})", sample_id)
    fig.subplots_adjust(hspace=0.60, wspace=0.35, top=0.88, bottom=0.16)

    bins = np.linspace(0, 1, 41)
    for idx, kar in enumerate(top_kars):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sub = mut[mut["karyotype"]==kar]["VAF"].dropna()
        colour = pal.get(kar, "#888888")
        ax.hist(sub, bins=bins, color=colour, alpha=0.72, edgecolor="white")

        major, minor = _parse_karyotype(kar)
        if not np.isnan(major):
            m1 = _expected_vaf(purity_f, major, minor, 1)
            ax.axvline(m1, color="black", lw=1.8, ls="--", label=f"m=1 -> {m1:.2f}")
            if major >= 2:
                m2 = _expected_vaf(purity_f, major, minor, 2)
                ax.axvline(m2, color="#8338EC", lw=1.8, ls=":", label=f"m=2 -> {m2:.2f}")
        ax.legend(fontsize=6.8, loc="upper right")
        _style(ax, f"{kar}  (n={len(sub)})", "VAF", "Count", fs=9.5)
    for idx in range(len(top_kars), nrows*ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    _caption(fig,
        "Dashed/dotted lines show the theoretical VAF a clonal mutation is expected at, "
        "given this sample's purity and the segment's copy number, for mutation "
        "multiplicity m=1 (mutation on one allele copy) or m=2 (present on two copies, "
        "i.e. it predates a copy gain). Formula: VAF = m*purity / (2*(1-purity) + purity*CN_total). "
        "If the histogram's peak lines up with a dashed/dotted line, purity and copy number "
        "are consistent with the data for that karyotype. If it doesn't, that karyotype's "
        "CN call or the global purity estimate may need revisiting.\n"
        "NOTE: this is a simplified illustrative reproduction of CNAqc's peak logic "
        "(single-mutation binomial expectation), not CNAqc's own binomial-mixture / "
        "entropy-based peak-matching algorithm. See the next page for CNAqc's own "
        "verdict if available.",
        y=0.01)
    _save(pdf, fig)


def page_official_peaks(pdf, peaks, sample_id):
    if peaks is None:
        _no_data_page(pdf, "CNAqc Official Peak-Matching QC", sample_id,
                      "peaks_matches.tsv was not found. This file is only produced if "
                      "CNAqc::analyze_peaks(x) was run on the object before saving. "
                      "To generate it, run in R:\n\n"
                      "  x <- CNAqc::analyze_peaks(x)\n"
                      "  saveRDS(x, 'sample_cnaqc_with_peaks.rds')\n\n"
                      "then re-run the R extractor on the updated RDS.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    _header(fig, "CNAqc Official Peak-Matching QC", sample_id)
    ax.axis("off")

    cols_show = [c for c in peaks.columns][:8]
    n_show = min(len(peaks), 20)
    rows = peaks[cols_show].head(n_show).astype(str).values.tolist()

    tbl = ax.table(cellText=rows, colLabels=cols_show, loc="center", cellLoc="center",
                   bbox=[0.02, 0.15, 0.96, 0.72])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for j in range(len(cols_show)):
        tbl[0,j].set_facecolor("#2C3E50")
        tbl[0,j].set_text_props(color="white", fontweight="bold")

    if len(peaks) > n_show:
        _caption(fig, f"Showing first {n_show} of {len(peaks)} rows from peaks_matches.tsv "
                      f"(CNAqc::analyze_peaks() output). This is the authoritative, statistically "
                      f"rigorous version of the illustrative comparison on the previous page.",
                y=0.05)
    else:
        _caption(fig, "peaks_matches.tsv (CNAqc::analyze_peaks() output) -- the authoritative "
                      "statistically rigorous version of the illustrative comparison on the "
                      "previous page.", y=0.05)
    _save(pdf, fig)


def page_segment_ccf(pdf, cna, sample_id):
    if cna is None or "CCF" not in cna.columns:
        _no_data_page(pdf, "CNA Segment Clonality (segment-level CCF)", sample_id,
                      "cna_clonal.tsv was not provided, or has no CCF column.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    _header(fig, "CNA Segment Clonality  (segment-level CCF)", sample_id)
    fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.35)

    ccf = cna["CCF"].dropna()
    ax1.hist(ccf, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax1.axvline(1.0, color="#E74C3C", ls="--", lw=1.3, label="CCF = 1.0 (fully clonal)")
    ax1.legend(fontsize=8)
    _style(ax1, "Distribution of Segment CCF", "Segment CCF (fraction of cells with this CNA)", "Segment count")

    n_clonal = (ccf >= 0.99).sum()
    n_sub    = (ccf < 0.99).sum()
    if n_clonal + n_sub > 0:
        ax2.pie([n_clonal, n_sub], labels=[f"Clonal CNA\n(CCF~1)\nn={n_clonal}",
                                            f"Subclonal CNA\n(CCF<1)\nn={n_sub}"],
                colors=["#55A868","#E63946"], autopct="%1.0f%%", startangle=90,
                textprops={"fontsize":9})
    ax2.set_title("Clonal vs Subclonal CNA Segments", fontsize=10, fontweight="bold", pad=8)

    _caption(fig,
        "IMPORTANT DISTINCTION: the 'CCF' column in the CNA segment table is the "
        "cellular prevalence of the copy-number ALTERATION itself -- i.e. what fraction "
        "of tumour cells carry that particular copy-number change. This is NOT the same "
        "as per-mutation CCF (the fraction of cells carrying an individual point mutation), "
        "which is a separate quantity computed by CNAqc::compute_CCF() -- see the next page. "
        "If every segment shows CCF=1 here, it means all detected copy-number alterations "
        "are clonal (present in the whole tumour), consistent with has_subclonal_CNA=FALSE "
        "in the metadata -- this sample shows no evidence of subclonal copy-number evolution.",
        y=0.01)
    _save(pdf, fig)


def page_mutation_ccf(pdf, mut, sample_id):
    if mut is None or "CCF" not in mut.columns:
        _no_data_page(pdf, "Per-mutation Cancer Cell Fraction (CCF)", sample_id,
                      "mutations.tsv has no CCF column. This means "
                      "CNAqc::compute_CCF(x) was not run on the object before saving. "
                      "To generate it, run in R:\n\n"
                      "  x <- CNAqc::compute_CCF(x)\n"
                      "  saveRDS(x, 'sample_cnaqc_with_ccf.rds')\n\n"
                      "then re-run the R extractor on the updated RDS.")
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _header(fig, "Per-mutation Cancer Cell Fraction (CCF)", sample_id)
    ccf = pd.to_numeric(mut["CCF"], errors="coerce").dropna()
    ax.hist(ccf, bins=40, color="#3BB273", alpha=0.8, edgecolor="white")
    ax.axvline(1.0, color="#E74C3C", ls="--", lw=1.3, label="CCF = 1.0 (clonal)")
    if len(ccf):
        ax.axvline(ccf.median(), color="black", ls=":", lw=1.3,
                   label=f"Median = {ccf.median():.2f}")
    ax.legend(fontsize=9)
    _style(ax, f"Per-mutation CCF Distribution  (n={len(ccf)})",
           "Cancer Cell Fraction (CCF)", "Mutation count")
    fig.subplots_adjust(bottom=0.18, top=0.88)
    _caption(fig,
        "CCF here is per-mutation: the estimated proportion of tumour cells carrying "
        "each individual point mutation, computed by CNAqc after phasing mutation "
        "multiplicity against the local copy-number state. Mutations near CCF=1 are "
        "clonal; a cluster of mutations below CCF=1 suggests subclonal structure.")
    _save(pdf, fig)


def page_vaf_vs_depth(pdf, mut, sample_id, pal):
    if mut is None or not {"VAF","DP","karyotype"}.issubset(mut.columns):
        _no_data_page(pdf, "VAF vs Depth", sample_id,
                      "mutations.tsv missing VAF, DP, or karyotype columns.")
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _header(fig, "VAF vs Read Depth  (coloured by karyotype)", sample_id)
    top_kars = mut["karyotype"].value_counts().head(8).index.tolist()
    for kar in top_kars:
        sub = mut[mut["karyotype"]==kar]
        ax.scatter(sub["DP"], sub["VAF"], color=pal.get(kar,"#888888"),
                   s=14, alpha=0.55, edgecolors="none", label=kar)
    ax.legend(fontsize=8, markerscale=1.5, loc="upper right", ncol=2)
    _style(ax, "VAF vs Read Depth", "Read depth (DP)", "VAF")
    fig.subplots_adjust(bottom=0.18, top=0.88)
    _caption(fig,
        "Each dot = one mutation, coloured by karyotype. Low-depth mutations (left side) "
        "have noisier VAF estimates and contribute more to peak-detection uncertainty. "
        "Consistent VAF regardless of depth within a karyotype supports a clean, "
        "well-estimated copy-number/purity model for that state.")
    _save(pdf, fig)


def page_genome_landscape(pdf, mut, cna, sample_id, pal):
    has_cna_coords = cna is not None and {"chr","start","end"}.issubset(cna.columns)
    has_mut_coords = mut is not None and {"chr","pos"}.issubset(mut.columns)
    if not has_cna_coords and not has_mut_coords:
        _no_data_page(pdf, "Genome-wide Landscape", sample_id,
                      "Neither CNA segments nor mutations have usable genomic coordinates.")
        return

    present = []
    if has_cna_coords:
        present = [c for c in CHR_ORDER if c in cna["chr"].dropna().unique()]
    elif has_mut_coords:
        present = [c for c in CHR_ORDER if c in mut["chr"].dropna().unique()]
    if not present:
        return

    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    _header(fig, "Genome-wide Karyotype and Mutation Landscape", sample_id)
    fig.subplots_adjust(hspace=0.10, bottom=0.13, top=0.90)

    if has_cna_coords:
        cna2 = cna[cna["chr"].isin(present)].copy()
        for _, seg in cna2.iterrows():
            major_v = seg.get("Major", np.nan)
            minor_v = seg.get("minor", np.nan)
            if pd.isna(major_v):
                kar = "?"
                total_cn = 0
            else:
                kar = f"{int(major_v)}:{int(minor_v) if pd.notna(minor_v) else 0}"
                total_cn = (major_v or 0) + (minor_v or 0)
            colour = pal.get(kar, "#888888")
            gx0 = offsets.get(seg["chr"],0) + seg["start"]
            gx1 = offsets.get(seg["chr"],0) + seg["end"]
            ax1.barh(0, gx1-gx0, left=gx0, height=min(total_cn/8,1)*0.7+0.15,
                     color=colour, alpha=0.75, edgecolor="none")
        ax1.set_ylim(-0.1,1.1); ax1.set_yticks([])
        ax1.set_title("Copy-number segments  (bar height ~ total CN, colour = karyotype)",
                      fontsize=10, fontweight="bold", pad=6)
    else:
        ax1.axis("off")
        ax1.text(0.5,0.5,"No CNA coordinates available", ha="center", transform=ax1.transAxes)

    if has_mut_coords:
        mut2 = mut[mut["chr"].isin(present)].copy()
        mut2["gx"] = mut2["chr"].map(offsets).fillna(0) + mut2["pos"].fillna(0)
        top_kars = mut2["karyotype"].value_counts().head(8).index.tolist() if "karyotype" in mut2.columns else []
        if top_kars:
            for kar in top_kars:
                sub = mut2[mut2["karyotype"]==kar]
                ax2.scatter(sub["gx"], sub["VAF"], color=pal.get(kar,"#888888"),
                           s=7, alpha=0.5, edgecolors="none", label=kar)
            ax2.legend(fontsize=7, loc="upper right", ncol=2, markerscale=1.5)
        else:
            ax2.scatter(mut2["gx"], mut2.get("VAF", pd.Series(dtype=float)),
                       color="#4C72B0", s=7, alpha=0.5, edgecolors="none")
        ax2.set_ylim(0,1.05)
        _style(ax2, "Mutation VAF across the Genome", "Genomic position", "VAF")
    else:
        ax2.axis("off")

    for ch in present:
        for ax in (ax1, ax2):
            ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch,100e6)/2
        ax2.text(mid, -0.07, ch.replace("chr",""), ha="center", fontsize=6,
                 color="#555555", transform=ax2.get_xaxis_transform())
    ax1.set_xlim(0, cum)

    _caption(fig,
        "Top: copy-number segments across the genome, bar height proportional to total "
        "copy number, colour = karyotype. Bottom: mutation VAF at each genomic position, "
        "coloured the same way. Regions where mutation VAF visibly shifts in step with the "
        "copy-number track are behaving as expected under the CNAqc model.",
        y=0.02)
    _save(pdf, fig)


def page_depth_distribution(pdf, mut, sample_id):
    if mut is None or "DP" not in mut.columns:
        _no_data_page(pdf, "Read Depth Distribution", sample_id,
                      "mutations.tsv missing DP column.")
        return

    dp = mut["DP"].dropna()
    fig, ax = plt.subplots(figsize=(9, 6))
    _header(fig, "Read Depth Distribution  (all mutations)", sample_id)
    p99 = np.percentile(dp, 99) if len(dp) else 1
    ax.hist(dp.clip(upper=p99), bins=50, color="#4C72B0", alpha=0.8, edgecolor="white")
    if len(dp):
        ax.axvline(dp.median(), color="black", ls="--", lw=1.4,
                   label=f"Median = {dp.median():.0f}x")
        ax.legend(fontsize=9)
    _style(ax, "Read Depth at Mutation Sites  (clipped at 99th percentile)",
           "Read depth (DP)", "Mutation count")
    fig.subplots_adjust(bottom=0.18, top=0.88)
    _caption(fig,
        "General sanity check: read depth at each mutation site. Very low depth "
        "mutations contribute the most noise to CNAqc's VAF-peak analysis; if this "
        "distribution is skewed unexpectedly low, investigate upstream capture/coverage.")
    _save(pdf, fig)

# main

def main():
    parser = argparse.ArgumentParser(
        description="Human-understandable PDF report for CNAqc output."
    )
    parser.add_argument("--dir", "-d", default=None,
                        help="Directory with mutations.tsv / cna_clonal.tsv / "
                             "karyotype_summary.tsv / metadata.json / peaks_matches.tsv "
                             "(as written by extract_cnaqc_rds.R)")
    parser.add_argument("--mutations",         default=None)
    parser.add_argument("--cna",               default=None, help="cna_clonal.tsv")
    parser.add_argument("--cna-subclonal",     default=None, dest="cna_subclonal")
    parser.add_argument("--karyotype-summary", default=None, dest="karyotype_summary")
    parser.add_argument("--metadata",          default=None)
    parser.add_argument("--peaks-matches",     default=None, dest="peaks_matches")
    parser.add_argument("--qc-txt",            default=None, dest="qc_txt",
                        help="Plain-text automated QC verdict file (tab-separated)")
    parser.add_argument("--sample", "-s", default="sample")
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    def _res(explicit, d, fname):
        if explicit: return explicit
        if d: return str(Path(d) / fname)
        return None

    mutations_p  = _res(args.mutations, args.dir, "mutations.tsv")
    cna_p        = _res(args.cna, args.dir, "cna_clonal.tsv")
    cna_sub_p    = _res(args.cna_subclonal, args.dir, "cna_subclonal.tsv")
    kar_p        = _res(args.karyotype_summary, args.dir, "karyotype_summary.tsv")
    meta_p       = _res(args.metadata, args.dir, "metadata.json")
    peaks_p      = _res(args.peaks_matches, args.dir, "peaks_matches.tsv")

    sid     = args.sample
    out_pdf = Path(args.output) if args.output else Path(f"{sid}_cnaqc_report.pdf")

    print(f"[cnaqc_report] Sample: {sid}")
    if args.dir:
        print(f"[cnaqc_report] Reading from directory: {args.dir}")

    meta        = load_metadata(meta_p)
    qc          = load_qc_txt(args.qc_txt)
    mut         = load_mutations(mutations_p)
    cna         = load_cna(cna_p, cna_sub_p)
    kar_summary = _safe_read_tsv(kar_p, "karyotype_summary.tsv")
    peaks       = _safe_read_tsv(peaks_p, "peaks_matches.tsv")

    if mut is None and cna is None:
        warn("Neither mutations nor CNA data could be loaded. "
             "The report will contain mostly placeholder pages.")

    all_kars = set()
    if mut is not None and "karyotype" in mut.columns:
        all_kars |= set(mut["karyotype"].dropna().unique())
    if cna is not None and {"Major","minor"}.issubset(cna.columns):
        all_kars |= set(
            (cna["Major"].astype("Int64").astype(str) + ":" +
             cna["minor"].astype("Int64").astype(str)).dropna().unique()
        )
    pal = _kar_palette(all_kars)

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.5,
    })

    print(f"[cnaqc_report] Writing: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        d = pdf.infodict()
        d["Title"]  = f"CNAqc Report - {sid}"
        d["Author"] = "cnaqc_report.py"

        page_overview(pdf, meta, qc, mut, cna, kar_summary, sid)
        page_qc_verdict(pdf, qc, sid)
        page_karyotype_segments(pdf, kar_summary, cna, sid, pal)
        page_karyotype_bp(pdf, kar_summary, sid, pal)
        page_mutations_per_karyotype(pdf, mut, sid, pal)
        page_depth_distribution(pdf, mut, sid)
        page_vaf_by_karyotype(pdf, mut, sid, pal)
        page_expected_vs_observed(pdf, mut, meta, sid, pal)
        page_official_peaks(pdf, peaks, sid)
        page_segment_ccf(pdf, cna, sid)
        page_mutation_ccf(pdf, mut, sid)
        page_vaf_vs_depth(pdf, mut, sid, pal)
        page_genome_landscape(pdf, mut, cna, sid, pal)

    print(f"[cnaqc_report] Done -- {out_pdf}")


if __name__ == "__main__":
    main()