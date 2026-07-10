#!/usr/bin/env python3
"""
deepsomatic_report.py — Visual QC report for DeepSomatic somatic VCF output.

Usage:
    python deepsomatic_report.py input.vcf.gz [--output report.pdf] [--sample SAMPLE_ID]

Conda dependencies (all available via conda-forge / bioconda):
    conda install -c conda-forge -c bioconda cyvcf2 matplotlib numpy pandas

One plot per PDF page; each page carries a short interpretation sentence.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

try:
    from cyvcf2 import VCF
except ImportError:
    sys.exit("ERROR: cyvcf2 not found. Install with: conda install -c bioconda cyvcf2")

# ── colour palette ────────────────────────────────────────────────────────────
C_SNV   = "#4C72B0"   # blue
C_INDEL = "#DD8452"   # orange
C_PASS  = "#55A868"   # green
C_LOW   = "#C44E52"   # red
C_MED   = "#8172B2"   # purple
C_ACC   = "#64B5CD"   # teal

SNV_PALETTE = {
    "C>A": "#3B9AB2", "C>G": "#78B7C5", "C>T": "#EBCC2A",
    "T>A": "#E1AF00", "T>C": "#F21A00", "T>G": "#CB4335",
}

CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]

# ── helper ────────────────────────────────────────────────────────────────────

def _annotate_caption(ax, text, fontsize=8.5):
    """Place an italic caption in a reserved strip below the axes."""
    fig = ax.figure
    pos = ax.get_position()
    fig.text(
        pos.x0, pos.y0 - 0.06,
        text, ha="left", va="top", fontsize=fontsize,
        fontstyle="italic", color="#444444",
        transform=fig.transFigure,
        wrap=True,
    )


def _style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _page_title(fig, title, sample_id):
    fig.suptitle(f"{title}  |  {sample_id}", fontsize=14, fontweight="bold", y=0.97)


def _bar_labels(ax, bars, fmt="{:.0f}", fontsize=8):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + h * 0.015,
                fmt.format(h), ha="center", va="bottom", fontsize=fontsize,
            )

# ── VCF parsing ───────────────────────────────────────────────────────────────

def parse_vcf(vcf_path):
    vcf = VCF(str(vcf_path))
    records = []

    for v in vcf:
        chrom = v.CHROM
        pos   = v.POS
        ref   = v.REF
        alts  = v.ALT
        qual  = v.QUAL if v.QUAL is not None else float("nan")
        filt  = v.FILTER or "PASS"   # cyvcf2 returns None for PASS

        # variant type
        alt = alts[0] if alts else "."
        if len(ref) == 1 and len(alt) == 1:
            vtype = "SNV"
        else:
            vtype = "INDEL"

        # FORMAT fields — grab from first sample
        gt   = v.genotypes[0]     # list [allele1, allele2, phased]
        fmt_dp  = _fmt_int(v, "DP")
        fmt_vaf = _fmt_float(v, "VAF")
        fmt_gq  = _fmt_int(v, "GQ")
        fmt_ndp = _fmt_int(v, "NDP")   # normal depth

        # indel length
        indel_len = len(alt) - len(ref)   # +ve ins, -ve del

        # SNV substitution class
        sub_class = None
        if vtype == "SNV":
            sub_class = _substitution_class(ref, alt)

        records.append({
            "chrom":     chrom,
            "pos":       pos,
            "ref":       ref,
            "alt":       alt,
            "qual":      qual,
            "filter_label": filt if filt else "PASS",
            "vtype":     vtype,
            "DP":        fmt_dp,
            "VAF":       fmt_vaf,
            "GQ":        fmt_gq,
            "NDP":       fmt_ndp,
            "indel_len": indel_len,
            "sub_class": sub_class,
        })

    vcf.close()
    return pd.DataFrame(records)


def _fmt_int(v, key):
    try:
        val = v.format(key)
        if val is not None and len(val) > 0:
            return int(val[0][0]) if hasattr(val[0], "__len__") else int(val[0])
    except Exception:
        pass
    return np.nan


def _fmt_float(v, key):
    try:
        val = v.format(key)
        if val is not None and len(val) > 0:
            return float(val[0][0]) if hasattr(val[0], "__len__") else float(val[0])
    except Exception:
        pass
    return np.nan


def _substitution_class(ref, alt):
    """Pyrimidine-normalised substitution class (e.g. C>T, T>A)."""
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    if ref in ("C", "T"):
        return f"{ref}>{alt}"
    else:
        return f"{comp.get(ref, ref)}>{comp.get(alt, alt)}"

# ── individual pages ──────────────────────────────────────────────────────────

def page_summary(pdf, df, sample_id):
    """Page 1 — high-level summary statistics as a styled table."""
    fig, ax = plt.subplots(figsize=(10, 7))
    _page_title(fig, "Summary Statistics", sample_id)
    ax.axis("off")

    snvs   = (df.vtype == "SNV").sum()
    indels = (df.vtype == "INDEL").sum()
    total  = len(df)
    pass_v = (df.filter_label == "PASS").sum()
    ts     = df[(df.vtype == "SNV") & (df.sub_class.isin(["C>T", "T>C"]))].shape[0]
    tv     = snvs - ts

    rows = [
        ["Total variants",            f"{total:,}"],
        ["SNVs",                       f"{snvs:,}"],
        ["INDELs",                     f"{indels:,}"],
        ["PASS variants",              f"{pass_v:,}  ({100*pass_v/max(total,1):.1f} %)"],
        ["Ts/Tv ratio (SNVs)",         f"{ts/max(tv,1):.3f}"],
        ["Median VAF",                 f"{df.VAF.median():.3f}" if df.VAF.notna().any() else "N/A"],
        ["Median tumour DP",           f"{df.DP.median():.0f}" if df.DP.notna().any() else "N/A"],
        ["Median normal DP (NDP)",     f"{df.NDP.median():.0f}" if df.NDP.notna().any() else "N/A"],
        ["Median GQ",                  f"{df.GQ.median():.0f}" if df.GQ.notna().any() else "N/A"],
    ]

    col_labels = ["Metric", "Value"]
    table = ax.table(
        cellText=rows, colLabels=col_labels,
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.4, 2.2)

    # colour header
    for j in range(2):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", fontweight="bold")
    # alternate row shading
    for i in range(1, len(rows) + 1):
        fc = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(fc)

    fig.text(0.12, 0.15,
        "Overview of key quality and quantity metrics derived from the DeepSomatic VCF. "
        "A Ts/Tv ratio between 2.0–3.5 is expected for exome somatic SNVs; "
        "values outside this range may indicate filtering artefacts.",
        fontsize=9, fontstyle="italic", color="#444444", wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_variant_counts(pdf, df, sample_id):
    """Page 2 — SNV vs INDEL counts, split by filter status."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _page_title(fig, "Variant Counts by Type and Filter Status", sample_id)
    fig.subplots_adjust(bottom=0.26)

    # left: SNV vs INDEL
    ax = axes[0]
    counts = df.vtype.value_counts().reindex(["SNV", "INDEL"], fill_value=0)
    bars = ax.bar(counts.index, counts.values, color=[C_SNV, C_INDEL], width=0.5, edgecolor="white")
    _bar_labels(ax, bars)
    _style_ax(ax, "SNVs vs INDELs", "Variant type", "Count")

    # right: stacked PASS vs other per type
    ax2 = axes[1]
    for vt, colour, x in [("SNV", C_SNV, 0), ("INDEL", C_INDEL, 1)]:
        sub = df[df.vtype == vt]
        n_pass  = (sub.filter_label == "PASS").sum()
        n_other = len(sub) - n_pass
        b1 = ax2.bar(x, n_pass,  color=C_PASS,   label="PASS"   if x == 0 else "", edgecolor="white")
        b2 = ax2.bar(x, n_other, bottom=n_pass, color=C_LOW, label="Non-PASS" if x == 0 else "", edgecolor="white")
        ax2.text(x, n_pass / 2,            f"{n_pass}",  ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax2.text(x, n_pass + n_other / 2,  f"{n_other}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["SNV", "INDEL"])
    ax2.legend(fontsize=9)
    _style_ax(ax2, "PASS vs Non-PASS (stacked)", "Variant type", "Count")

    for ax in axes:
        _annotate_caption(ax,
            "SNVs are single-base substitutions; INDELs are insertions or deletions. "
            "Only PASS variants are used in downstream somatic analyses."
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_chromosomes(pdf, df, sample_id):
    """Page 3 — variant counts per chromosome."""
    fig, ax = plt.subplots(figsize=(14, 6))
    _page_title(fig, "Variant Distribution Across Chromosomes", sample_id)
    fig.subplots_adjust(bottom=0.30)

    present = [c for c in CHR_ORDER if c in df.chrom.values]
    missing = sorted(set(df.chrom.unique()) - set(CHR_ORDER))
    order   = present + missing

    snv_c   = df[df.vtype == "SNV"].groupby("chrom").size().reindex(order, fill_value=0)
    indel_c = df[df.vtype == "INDEL"].groupby("chrom").size().reindex(order, fill_value=0)

    x = np.arange(len(order))
    w = 0.4
    ax.bar(x - w/2, snv_c,   width=w, label="SNV",   color=C_SNV,   edgecolor="white")
    ax.bar(x + w/2, indel_c, width=w, label="INDEL", color=C_INDEL, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    _style_ax(ax, "Somatic Variants per Chromosome", "Chromosome", "Variant count")
    _annotate_caption(ax,
        "Uneven distribution across autosomes is expected and reflects chromosome length. "
        "Elevated counts on sex chromosomes may warrant closer inspection in paediatric tumours."
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_snv_spectrum(pdf, df, sample_id):
    """Page 4 — SNV substitution spectrum + Ts/Tv breakdown."""
    snvs = df[df.vtype == "SNV"].dropna(subset=["sub_class"])
    if snvs.empty:
        return

    # Ts = C>T and T>C (transitions); everything else = Tv
    TS_CLASSES = {"C>T", "T>C"}
    TV_CLASSES = {"C>A", "C>G", "T>A", "T>G"}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2, 1]})
    _page_title(fig, "SNV Substitution Spectrum & Ts/Tv Ratio", sample_id)
    fig.subplots_adjust(bottom=0.28, wspace=0.35)

    # ── left: spectrum bar chart ───────────────────────────────────────────
    counts = snvs.sub_class.value_counts().reindex(SNV_PALETTE.keys(), fill_value=0)
    colours = [SNV_PALETTE[k] for k in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=0.8)
    _bar_labels(ax, bars)

    total_snv = snvs.shape[0]
    for bar, val in zip(bars, counts.values):
        pct = 100 * val / max(total_snv, 1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total_snv * 0.005,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, color="#333333")

    _style_ax(ax, "Substitution Classes (pyrimidine-normalised)", "Substitution", "Count")
    _annotate_caption(ax,
        "Pyrimidine-normalised spectrum (C/T as reference). C>T transitions dominate in "
        "age-related (SBS1/SBS5) and UV-induced (SBS7) processes. Elevated C>A suggests "
        "oxidative damage (SBS18)."
    )

    # ── right: Ts / Tv stacked bar + ratio annotation ─────────────────────
    n_ts = snvs[snvs.sub_class.isin(TS_CLASSES)].shape[0]
    n_tv = snvs[snvs.sub_class.isin(TV_CLASSES)].shape[0]
    ratio = n_ts / max(n_tv, 1)

    ts_breakdown = {k: counts.get(k, 0) for k in TS_CLASSES}
    tv_breakdown = {k: counts.get(k, 0) for k in TV_CLASSES}

    # stacked bar: Ts classes stacked green tones, Tv classes stacked orange tones
    ts_colours = ["#27AE60", "#52BE80"]
    tv_colours = ["#E67E22", "#F0B27A", "#D35400", "#F39C12"]

    bottom = 0
    for (label, val), col in zip(ts_breakdown.items(), ts_colours):
        ax2.bar(0, val, bottom=bottom, color=col, edgecolor="white", width=0.5, label=f"Ts: {label}")
        if val > 0:
            ax2.text(0, bottom + val / 2, f"{label}\n{val}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
        bottom += val

    for (label, val), col in zip(tv_breakdown.items(), tv_colours):
        ax2.bar(0, val, bottom=bottom, color=col, edgecolor="white", width=0.5, label=f"Tv: {label}")
        if val > 0:
            ax2.text(0, bottom + val / 2, f"{label}\n{val}", ha="center", va="center",
                     fontsize=8, color="white", fontweight="bold")
        bottom += val

    ax2.set_xticks([0])
    ax2.set_xticklabels(["SNVs"])
    ax2.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.55, 1.0))

    # annotate ratio prominently
    ax2.set_title("Ts / Tv Breakdown", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Count", fontsize=10)
    ax2.set_xlabel("", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(labelsize=9)

    # ratio text box
    ax2.text(0, bottom * 1.04,
             f"Ts/Tv = {ratio:.3f}",
             ha="center", va="bottom", fontsize=11, fontweight="bold",
             color="#2C3E50",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#EBF5FB", edgecolor="#2C3E50", linewidth=1.2))

    ax2.set_ylim(0, bottom * 1.18)

    _annotate_caption(ax2,
        "Ts = transitions (C>T, T>C); Tv = transversions (C>A, C>G, T>A, T>G). "
        "Expected Ts/Tv for WES somatic SNVs is ~2.0–3.5."
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vaf(pdf, df, sample_id):
    """Page 5 — VAF distribution, split by variant type."""
    sub = df.dropna(subset=["VAF"])
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _page_title(fig, "Variant Allele Frequency (VAF) Distribution", sample_id)
    fig.subplots_adjust(bottom=0.26)

    bins = np.linspace(0, 1, 41)

    # left: all variants histogram
    ax = axes[0]
    ax.hist(sub[sub.vtype == "SNV"].VAF,   bins=bins, color=C_SNV,   alpha=0.75, label="SNV",   edgecolor="white")
    ax.hist(sub[sub.vtype == "INDEL"].VAF, bins=bins, color=C_INDEL, alpha=0.75, label="INDEL", edgecolor="white")
    ax.axvline(sub.VAF.median(), color="black", ls="--", lw=1.2, label=f"Median={sub.VAF.median():.2f}")
    ax.legend(fontsize=9)
    _style_ax(ax, "VAF Distribution (all variants)", "VAF", "Count")

    # right: PASS-only box+strip per type
    ax2 = axes[1]
    pass_data = sub[sub.filter_label == "PASS"]
    for i, (vt, col) in enumerate([("SNV", C_SNV), ("INDEL", C_INDEL)]):
        vdata = pass_data[pass_data.vtype == vt].VAF.dropna()
        if vdata.empty:
            continue
        bp = ax2.boxplot(vdata, positions=[i], widths=0.4, patch_artist=True,
                         boxprops=dict(facecolor=col, alpha=0.6),
                         medianprops=dict(color="black", lw=2),
                         flierprops=dict(marker=".", markersize=3, alpha=0.4, color=col))
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["SNV", "INDEL"])
    ax2.set_ylim(0, 1.05)
    _style_ax(ax2, "VAF Boxplot (PASS only)", "Variant type", "VAF")

    for ax in axes:
        _annotate_caption(ax,
            "VAF is the fraction of reads supporting the alternate allele in the tumour. "
            "Low-VAF clusters may represent subclonal mutations; VAF ~0.5 or ~1.0 "
            "can indicate clonal events or copy-number-driven artefacts."
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_depth(pdf, df, sample_id):
    """Page 6 — Read depth (tumour DP and normal NDP) distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _page_title(fig, "Read Depth Distribution", sample_id)
    fig.subplots_adjust(bottom=0.26)

    for ax, col, label, colour in [
        (axes[0], "DP",  "Tumour read depth (DP)",  C_SNV),
        (axes[1], "NDP", "Normal read depth (NDP)", C_ACC),
    ]:
        data = df[col].dropna()
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        p99 = np.percentile(data, 99)
        clipped = data.clip(upper=p99)
        ax.hist(clipped, bins=50, color=colour, edgecolor="white", alpha=0.85)
        ax.axvline(data.median(), color="black", ls="--", lw=1.2,
                   label=f"Median = {data.median():.0f}×")
        ax.axvline(data.mean(),   color="#E74C3C", ls=":",  lw=1.2,
                   label=f"Mean = {data.mean():.0f}×")
        ax.legend(fontsize=9)
        _style_ax(ax, label, "Read depth (×)", "Variant count")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}×"))
        _annotate_caption(ax,
            "Sufficient coverage is critical for reliable somatic calling. "
            "DeepSomatic recommends ≥30× tumour and ≥25× normal depth for WES. "
            "Tail clipped at 99th percentile for display."
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_qual_gq(pdf, df, sample_id):
    """Page 7 — QUAL score and genotype quality (GQ) distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    _page_title(fig, "Variant Quality Scores", sample_id)
    fig.subplots_adjust(bottom=0.26)

    # QUAL
    ax = axes[0]
    qual = df.qual.dropna()
    if not qual.empty:
        p99 = np.percentile(qual, 99)
        ax.hist(qual.clip(upper=p99), bins=50, color=C_MED, edgecolor="white", alpha=0.85)
        ax.axvline(qual.median(), color="black", ls="--", lw=1.2, label=f"Median = {qual.median():.1f}")
        ax.legend(fontsize=9)
    _style_ax(ax, "QUAL Score Distribution", "QUAL", "Variant count")
    _annotate_caption(ax,
        "QUAL reflects DeepSomatic's confidence in the variant call. "
        "Low-QUAL variants (< 20) are more likely to be false positives."
    )

    # GQ
    ax2 = axes[1]
    gq = df.GQ.dropna()
    if not gq.empty:
        ax2.hist(gq, bins=50, color=C_ACC, edgecolor="white", alpha=0.85)
        ax2.axvline(gq.median(), color="black", ls="--", lw=1.2, label=f"Median = {gq.median():.0f}")
        ax2.legend(fontsize=9)
    _style_ax(ax2, "Genotype Quality (GQ) Distribution", "GQ", "Variant count")
    _annotate_caption(ax2,
        "GQ is the phred-scaled conditional probability that the assigned genotype is wrong. "
        "Higher GQ indicates more reliable genotype assignments."
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_indel_lengths(pdf, df, sample_id):
    """Page 8 — INDEL length distribution."""
    indels = df[df.vtype == "INDEL"]
    if indels.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    _page_title(fig, "INDEL Length Distribution", sample_id)
    fig.subplots_adjust(bottom=0.28)

    lengths = indels.indel_len
    max_abs = min(int(lengths.abs().quantile(0.99)) + 1, 30)
    bins = np.arange(-max_abs - 0.5, max_abs + 1.5, 1)

    ins_data = lengths[lengths > 0]
    del_data = lengths[lengths < 0]

    ax.hist(ins_data, bins=bins, color=C_SNV,   alpha=0.8, label=f"Insertions (n={len(ins_data)})", edgecolor="white")
    ax.hist(del_data, bins=bins, color=C_INDEL, alpha=0.8, label=f"Deletions  (n={len(del_data)})", edgecolor="white")
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    _style_ax(ax, "INDEL Length Distribution", "INDEL length (bp, + = insertion, − = deletion)", "Count")
    _annotate_caption(ax,
        "Positive values are insertions; negative values are deletions. "
        "Single-base INDELs in homopolymer runs are common sequencing artefacts in WES data."
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_substitution_matrix(pdf, df, sample_id):
    """New page — raw REF→ALT substitution counts: for each REF base, how many SNVs
    resulted in each possible ALT base (excluding the REF itself)."""
    snvs = df[(df.vtype == "SNV") & df.ref.isin(list("ACGT")) & df.alt.isin(list("ACGT"))].copy()
    if snvs.empty:
        return

    bases   = ["A", "C", "G", "T"]
    alt_map = {b: [x for x in bases if x != b] for b in bases}

    # colour scheme per ALT base (consistent across panels)
    ALT_COLOURS = {"A": "#E74C3C", "C": "#3498DB", "G": "#2ECC71", "T": "#F39C12"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _page_title(fig, "SNV Raw Substitution Matrix (REF → ALT)", sample_id)
    fig.subplots_adjust(bottom=0.14, hspace=0.55, wspace=0.35)

    for ax, ref_base in zip(axes.flat, bases):
        sub = snvs[snvs.ref == ref_base]
        total_ref = len(sub)

        alts    = alt_map[ref_base]
        counts  = [int((sub.alt == a).sum()) for a in alts]
        colours = [ALT_COLOURS[a] for a in alts]

        bars = ax.bar(
            [f"{ref_base}>{a}" for a in alts],
            counts, color=colours, edgecolor="white", linewidth=0.8, width=0.55,
        )

        for bar, val in zip(bars, counts):
            pct = 100 * val / max(total_ref, 1)
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    f"{val}\n({pct:.1f}%)",
                    ha="center", va="bottom", fontsize=8.5, color="#222222")

        ax.set_title(f"REF = {ref_base}  (n = {total_ref:,} SNVs)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Substitution", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        if total_ref == 0:
            ax.text(0.5, 0.5, "No variants", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")

    fig.text(0.05, 0.06,
        "Each panel shows, for a given reference base, how many somatic SNVs changed to each "
        "possible alternate base. Unlike the pyrimidine-collapsed spectrum, these counts are "
        "strand-aware and reflect the actual sequenced base. Dominant diagonals within a REF panel "
        "highlight the most frequent mutational event at that base.",
        fontsize=8.5, fontstyle="italic", color="#444444", wrap=True,
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_vaf_vs_depth(pdf, df, sample_id):
    """Page 10 — VAF vs depth scatter, coloured by variant type."""
    sub = df.dropna(subset=["VAF", "DP"])
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    _page_title(fig, "VAF vs. Read Depth", sample_id)
    fig.subplots_adjust(bottom=0.26)

    for ax, vt, col in [(axes[0], "SNV", C_SNV), (axes[1], "INDEL", C_INDEL)]:
        data = sub[sub.vtype == vt]
        if data.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        p99_dp = np.percentile(data.DP, 99)
        sc = ax.scatter(
            data.DP.clip(upper=p99_dp), data.VAF,
            c=data.qual, cmap="plasma", s=8, alpha=0.5, vmin=0, vmax=60,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("QUAL", fontsize=8)
        _style_ax(ax, f"{vt}: VAF vs Tumour Depth", "Tumour read depth (×)", "VAF")
        _annotate_caption(ax,
            "High-confidence calls (warm colour = high QUAL) should cluster at higher depths. "
            "Low-VAF calls at low depth are the most likely false positives."
        )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def page_filter_breakdown(pdf, df, sample_id):
    """Page 10 — Filter label breakdown."""
    fig, ax = plt.subplots(figsize=(9, 6))
    _page_title(fig, "Filter Label Breakdown", sample_id)
    fig.subplots_adjust(bottom=0.26)

    counts = df.filter_label.value_counts()
    colours = [C_PASS if f == "PASS" else C_LOW for f in counts.index]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colours[::-1], edgecolor="white")
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + counts.values.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    _style_ax(ax, "Variant Filter Labels", "Count", "Filter")
    _annotate_caption(ax,
        "GERMLINE filter flags inherited variants not removed prior to somatic calling. "
        "LowQual indicates calls below DeepSomatic's confidence threshold. "
        "Only PASS variants should be carried into downstream clonal analyses."
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate a multi-page visual QC report from a DeepSomatic VCF."
    )
    parser.add_argument("vcf", help="Path to DeepSomatic VCF (can be .vcf.gz)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output PDF path (default: <vcf_stem>_report.pdf)")
    parser.add_argument("--sample", "-s", default=None,
                        help="Sample ID shown in plot titles (default: inferred from filename)")
    args = parser.parse_args()

    vcf_path = Path(args.vcf)
    if not vcf_path.exists():
        sys.exit(f"ERROR: VCF not found: {vcf_path}")

    # derive sample ID and output path
    stem = vcf_path.name.replace(".vcf.gz", "").replace(".vcf", "")
    sample_id = args.sample or stem
    out_pdf   = Path(args.output) if args.output else vcf_path.parent / f"{stem}_report.pdf"

    print(f"[deepsomatic_report] Parsing VCF: {vcf_path}")
    df = parse_vcf(vcf_path)
    print(f"[deepsomatic_report] Loaded {len(df):,} variants")

    if df.empty:
        sys.exit("ERROR: No variants found in VCF.")

    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E0E0E0",
        "grid.linewidth":   0.6,
    })

    print(f"[deepsomatic_report] Writing report: {out_pdf}")
    with PdfPages(out_pdf) as pdf:
        # metadata
        d = pdf.infodict()
        d["Title"]   = f"DeepSomatic QC Report — {sample_id}"
        d["Author"]  = "deepsomatic_report.py"
        d["Subject"] = "Somatic variant QC"

        page_summary(pdf, df, sample_id)              # p1
        page_variant_counts(pdf, df, sample_id)       # p2
        page_chromosomes(pdf, df, sample_id)          # p3
        page_snv_spectrum(pdf, df, sample_id)         # p4 (spectrum + Ts/Tv)
        page_substitution_matrix(pdf, df, sample_id)  # p5 (raw REF→ALT matrix)
        page_vaf(pdf, df, sample_id)                  # p6
        page_depth(pdf, df, sample_id)                # p7
        page_qual_gq(pdf, df, sample_id)              # p8
        page_indel_lengths(pdf, df, sample_id)        # p9
        page_vaf_vs_depth(pdf, df, sample_id)         # p10
        page_filter_breakdown(pdf, df, sample_id)     # p11

    print(f"[deepsomatic_report] Done — {out_pdf}")


if __name__ == "__main__":
    main()