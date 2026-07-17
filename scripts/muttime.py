#!/usr/bin/env python3
"""
muttime_classify.py — Somatic mutation timing classification.

Implements the deterministic timing logic from Gerstung et al. Nature 2020
(PCAWG Evolution Working Group, Fig. 1b) using local allele-specific copy
number (FACETS) and Cancer Cell Fraction estimates (PyClone6 or VIBER) to
classify each somatic SNV as:

    early clonal    — present before copy-number gain (mutation copy number > 1)
    late clonal     — present after CN change on LOH segment (minor_cn = 0, total_cn > 2)
    clonal [NA]     — clonal, timing unresolvable from CN alone
    subclonal       — CCF < clonal threshold, not present in all tumour cells

Outputs (written before any plotting):
    <prefix>_mutations.tsv   — per-mutation timing + CCF + CN annotation
    <prefix>_segments.tsv    — per-segment summary (n muts, dominant CLS)
    <prefix>_summary.tsv     — CLS counts and fractions

Then:
    <prefix>_timing_report.pdf — one plot per page, overview + detailed panels

Usage:
    python muttime_classify.py \\
        --vcf     sg070_somatic_pass.vcf.gz \\
        --seg     sg070_cnv_segments.tsv \\
        --purity  sg070_purity.txt \\
        [--clusters  sg070_pyclone6_results.tsv] \\
        [--sample    sg070] \\
        [--prefix    results/sg070/muttime/sg070] \\
        [--clonal-threshold  0.80] \\
        [--subclonal-threshold  0.20]

Conda deps (all conda-forge / bioconda):
    conda install -c conda-forge -c bioconda cyvcf2 pandas numpy matplotlib scipy
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

try:
    from cyvcf2 import VCF
except ImportError:
    sys.exit("ERROR: cyvcf2 not found.  conda install -c bioconda cyvcf2")

# ── constants ─────────────────────────────────────────────────────────────────

CLS_COLOURS = {
    "early clonal":  "#2E86AB",   # blue
    "late clonal":   "#8338EC",   # purple
    "clonal [NA]":   "#3BB273",   # green
    "subclonal":     "#E63946",   # red
    "unclassified":  "#AAAAAA",   # grey
}
CLS_ORDER = ["early clonal", "late clonal", "clonal [NA]", "subclonal", "unclassified"]

CHR_ORDER = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
CHR_LENGTHS = {
    "chr1":249e6,"chr2":242e6,"chr3":198e6,"chr4":190e6,"chr5":182e6,
    "chr6":171e6,"chr7":159e6,"chr8":145e6,"chr9":138e6,"chr10":134e6,
    "chr11":135e6,"chr12":133e6,"chr13":115e6,"chr14":107e6,"chr15":102e6,
    "chr16":90e6,"chr17":83e6,"chr18":80e6,"chr19":59e6,"chr20":63e6,
    "chr21":48e6,"chr22":51e6,"chrX":156e6,"chrY":58e6,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def warn(msg):
    print(f"  [WARN] {msg}", file=sys.stderr)

def info(msg):
    print(f"  [INFO] {msg}", file=sys.stderr)


def _normalise_chrom(s):
    """Return chromosome name with 'chr' prefix."""
    s = str(s).strip()
    return s if s.startswith("chr") else f"chr{s}"


def _caption(fig, text, y=0.03, fontsize=8.0):
    fig.text(0.04, y, text, ha="left", va="bottom", fontsize=fontsize,
             fontstyle="italic", color="#444444", wrap=True,
             transform=fig.transFigure)


def _style(ax, title, xlabel, ylabel, fs=11):
    ax.set_title(title, fontsize=fs, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _cls_legend(ax, cls_present=None):
    patches = [
        mpatches.Patch(color=CLS_COLOURS[c], label=c)
        for c in CLS_ORDER
        if cls_present is None or c in cls_present
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper right")


def _save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_purity(path):
    try:
        val = float(Path(path).read_text().strip().split()[0])
        if not (0 < val <= 1):
            warn(f"Purity {val} outside (0,1] — clamping to 0.05–0.99")
            val = float(np.clip(val, 0.05, 0.99))
        info(f"Tumour purity = {val:.4f}")
        return val
    except Exception as e:
        warn(f"Could not read purity from {path}: {e} — assuming 1.0")
        return 1.0


def load_vcf(path):
    """Load VCF into DataFrame. Extracts AD, DP, VAF from FORMAT fields."""
    records = []
    try:
        vcf = VCF(str(path))
    except Exception as e:
        warn(f"Could not open VCF {path}: {e}")
        return pd.DataFrame()

    for v in vcf:
        chrom = _normalise_chrom(v.CHROM)
        pos   = v.POS
        ref   = v.REF
        alt   = v.ALT[0] if v.ALT else "."
        filt  = v.FILTER or "PASS"
        qual  = v.QUAL if v.QUAL is not None else np.nan

        # only SNVs for timing (indels are fine to keep but CN assignment noisier)
        is_snv = (len(ref) == 1 and len(alt) == 1)

        t_ref, t_alt, depth, vaf = np.nan, np.nan, np.nan, np.nan
        try:
            ad = v.format("AD")
            if ad is not None:
                t_ref = float(ad[0][0]); t_alt = float(ad[0][1])
                depth = t_ref + t_alt
                vaf   = t_alt / depth if depth > 0 else np.nan
        except Exception:
            pass

        if np.isnan(vaf):
            try:
                vaf_f = v.format("VAF")
                if vaf_f is not None:
                    vaf = float(vaf_f[0][0])
            except Exception:
                pass

        if np.isnan(depth):
            try:
                dp = v.format("DP")
                if dp is not None:
                    depth = float(dp[0][0])
                    if not np.isnan(vaf) and not np.isnan(depth):
                        t_alt = vaf * depth
                        t_ref = depth - t_alt
            except Exception:
                pass

        records.append(dict(
            chrom=chrom, pos=pos, ref=ref, alt=alt,
            filter=filt, qual=qual, is_snv=is_snv,
            t_ref=t_ref, t_alt=t_alt, depth=depth, vaf=vaf,
            mutation_id=f"{chrom}:{pos}:{ref}>{alt}",
        ))

    vcf.close()
    df = pd.DataFrame(records)
    info(f"VCF: {len(df)} total variants  "
         f"({df['is_snv'].sum()} SNVs, "
         f"{(~df['is_snv']).sum()} indels)")
    n_pass = (df["filter"] == "PASS").sum() if len(df) else 0
    info(f"VCF: {n_pass} PASS variants")
    if len(df) == 0:
        warn("VCF contains no variants — output will be empty")
    return df


def load_segments(path):
    """
    Load FACETS CN segment TSV.  Accepts the nA/nB or tcn/lcn column layouts
    written by build_phylogic_input2.R or raw FACETS output.
    """
    try:
        seg = pd.read_csv(path, sep="\t")
    except Exception as e:
        warn(f"Could not read segments {path}: {e}")
        return pd.DataFrame()

    seg["chrom"] = seg["chrom"].apply(_normalise_chrom) if "chrom" in seg.columns \
        else seg.iloc[:, 0].apply(_normalise_chrom)

    # detect column layout
    if "nA" in seg.columns and "nB" in seg.columns:
        seg = seg.rename(columns={"nA": "major_cn", "nB": "minor_cn"})
    elif "tcn" in seg.columns and "lcn" in seg.columns:
        seg = seg.rename(columns={"tcn": "major_cn", "lcn": "minor_cn"})
    elif "major_cn" not in seg.columns:
        warn(f"Segment file has no recognisable major/minor CN columns "
             f"(found: {list(seg.columns)}) — CN classification disabled")
        return pd.DataFrame()

    if "cf" not in seg.columns and "cf.em" in seg.columns:
        seg = seg.rename(columns={"cf.em": "cf"})
    if "cf" not in seg.columns:
        seg["cf"] = 1.0   # assume fully clonal if not present

    for col in ("major_cn", "minor_cn"):
        seg[col] = pd.to_numeric(seg[col], errors="coerce")
    seg["cf"] = pd.to_numeric(seg["cf"], errors="coerce").fillna(1.0)

    # detect start/end columns
    for s_col in ("start", "loc.start", "START"):
        if s_col in seg.columns:
            seg = seg.rename(columns={s_col: "start"}); break
    for e_col in ("end", "loc.end", "END"):
        if e_col in seg.columns:
            seg = seg.rename(columns={e_col: "end"}); break

    seg["start"] = pd.to_numeric(seg["start"], errors="coerce")
    seg["end"]   = pd.to_numeric(seg["end"],   errors="coerce")
    seg = seg.dropna(subset=["start","end","major_cn","minor_cn"])
    seg["major_cn"] = seg["major_cn"].astype(int)
    seg["minor_cn"] = seg["minor_cn"].clip(lower=0).astype(int)
    seg["total_cn"] = seg["major_cn"] + seg["minor_cn"]

    info(f"Segments: {len(seg)} rows across "
         f"{seg['chrom'].nunique()} chromosomes")

    sanity_issues = 0
    if (seg["minor_cn"] > seg["major_cn"]).any():
        warn("Some segments have minor_cn > major_cn — swapping")
        mask = seg["minor_cn"] > seg["major_cn"]
        seg.loc[mask, ["major_cn","minor_cn"]] = (
            seg.loc[mask, ["minor_cn","major_cn"]].values)
        sanity_issues += 1
    if (seg["total_cn"] == 0).any():
        n_del = (seg["total_cn"] == 0).sum()
        warn(f"{n_del} homozygous-deletion segments (total_cn=0) — "
             f"mutations on these will be unclassified")
    info(f"Segment sanity checks: {sanity_issues} issues found and corrected")
    return seg


def load_clusters(path):
    """
    Load PyClone6 results.tsv or VIBER mutations.tsv / parameters.tsv.
    Returns a DataFrame with columns: mutation_id, ccf
    """
    if path is None or not Path(path).exists():
        return None

    try:
        cl = pd.read_csv(path, sep="\t")
    except Exception as e:
        warn(f"Could not read clusters {path}: {e}")
        return None

    # ── PyClone6 results.tsv ──────────────────────────────────────────────
    if "cellular_prevalence" in cl.columns and "mutation_id" in cl.columns:
        cl["ccf"] = pd.to_numeric(cl["cellular_prevalence"], errors="coerce")
        info(f"Clusters: PyClone6 format, {len(cl)} mutations, "
             f"{cl['cluster_id'].nunique() if 'cluster_id' in cl.columns else '?'} clusters")
        return cl[["mutation_id","ccf"]].dropna()

    # ── VIBER mutations.tsv ───────────────────────────────────────────────
    if "cluster" in cl.columns and "vaf" in cl.columns and "mutation_id" in cl.columns:
        cl["ccf"] = pd.to_numeric(cl["vaf"], errors="coerce") * 2   # θ → CCF approx
        cl["ccf"] = cl["ccf"].clip(upper=1.0)
        info(f"Clusters: VIBER mutations.tsv format, {len(cl)} mutations")
        return cl[["mutation_id","ccf"]].dropna()

    warn(f"Could not recognise cluster file format "
         f"(columns: {list(cl.columns)[:8]}) — CCF will be estimated from VAF")
    return None

# ── segment assignment ────────────────────────────────────────────────────────

def assign_segments(muts, seg):
    """
    Assign each mutation to a CN segment by overlap.
    Adds: major_cn, minor_cn, total_cn, seg_cf columns.
    O(n_muts) via chromosome-wise sorted interval lookup.
    """
    if seg.empty:
        warn("No segments available — CN columns will be NaN")
        for col in ("major_cn","minor_cn","total_cn","seg_cf"):
            muts[col] = np.nan
        return muts

    results = {
        "major_cn": np.full(len(muts), np.nan),
        "minor_cn": np.full(len(muts), np.nan),
        "total_cn": np.full(len(muts), np.nan),
        "seg_cf":   np.full(len(muts), np.nan),
    }

    for chrom, group in muts.groupby("chrom"):
        s = seg[seg["chrom"] == chrom].sort_values("start")
        if s.empty:
            continue
        starts = s["start"].values
        ends   = s["end"].values
        for idx, row in group.iterrows():
            p = row["pos"]
            # binary search: find rightmost segment starting ≤ pos
            lo, hi = 0, len(starts) - 1
            found  = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if starts[mid] <= p:
                    found = mid; lo = mid + 1
                else:
                    hi = mid - 1
            if found >= 0 and ends[found] >= p:
                si = s.index[found]
                results["major_cn"][muts.index.get_loc(idx)] = s.at[si, "major_cn"]
                results["minor_cn"][muts.index.get_loc(idx)] = s.at[si, "minor_cn"]
                results["total_cn"][muts.index.get_loc(idx)] = s.at[si, "total_cn"]
                results["seg_cf"][muts.index.get_loc(idx)]   = s.at[si, "cf"]

    for col, vals in results.items():
        muts[col] = vals

    n_assigned = muts["major_cn"].notna().sum()
    n_total    = len(muts)
    pct        = 100 * n_assigned / max(n_total, 1)
    info(f"Segment assignment: {n_assigned}/{n_total} mutations assigned ({pct:.1f}%)")
    if pct < 50:
        warn("Less than 50% of mutations were assigned to CN segments. "
             "Check chromosome name style (chr1 vs 1) in VCF and segment file.")
    return muts

# ── mutation copy number + timing ─────────────────────────────────────────────

def compute_mut_copy_number(vaf, purity, total_cn, seg_cf=1.0):
    """
    Expected mutation copy number (m) given:
        m = VAF * (purity * total_cn * seg_cf + 2*(1-purity)) / purity
    Returns NaN if inputs are invalid.
    Reference: Gerstung et al. 2020 Supplementary Methods.
    """
    try:
        vaf       = float(vaf)
        purity    = float(purity)
        total_cn  = float(total_cn)
        seg_cf    = float(seg_cf) if not np.isnan(float(seg_cf)) else 1.0
        if np.isnan(vaf) or vaf <= 0 or purity <= 0 or total_cn <= 0:
            return np.nan
        denom = purity * total_cn * seg_cf + 2 * (1 - purity)
        if denom <= 0:
            return np.nan
        return vaf * denom / purity
    except Exception:
        return np.nan


def classify_timing(row, clonal_thr, subclonal_thr):
    """
    Gerstung 2020 deterministic timing rules:

    1. CCF < subclonal_thr  → subclonal
    2. CCF >= clonal_thr AND mut_copy_number > 1.0  → early clonal
       (mutation was co-amplified; present on >1 allele copy)
    3. CCF >= clonal_thr AND total_cn > 2 AND minor_cn == 0  → late clonal
       (LOH segment; mutation on the surviving allele)
    4. CCF >= clonal_thr, timing not resolvable from CN  → clonal [NA]
    5. No CCF/CN data  → unclassified
    """
    ccf      = row.get("ccf",           np.nan)
    mcn      = row.get("mut_copy_number", np.nan)
    total_cn = row.get("total_cn",      np.nan)
    minor_cn = row.get("minor_cn",      np.nan)

    if np.isnan(ccf):
        return "unclassified"

    if ccf < subclonal_thr:
        return "subclonal"

    if ccf >= clonal_thr:
        if not np.isnan(mcn) and mcn > 1.2:          # tolerance for noise
            return "early clonal"
        if (not np.isnan(total_cn) and not np.isnan(minor_cn)
                and total_cn > 2 and minor_cn == 0):
            return "late clonal"
        return "clonal [NA]"

    # between thresholds — borderline subclonal
    return "subclonal"


def run_classification(muts, purity, clusters_df, clonal_thr, subclonal_thr):
    """Add ccf, mut_copy_number, CLS to muts DataFrame."""

    # ── attach CCF ────────────────────────────────────────────────────────
    if clusters_df is not None and not clusters_df.empty:
        muts = muts.merge(clusters_df, on="mutation_id", how="left")
        n_ccf = muts["ccf"].notna().sum()
        info(f"CCF matched: {n_ccf}/{len(muts)} mutations from cluster file")
        if n_ccf == 0:
            warn("No mutation IDs matched between VCF and cluster file. "
                 "Check that mutation_id format matches (chr:pos:ref>alt).")
    else:
        info("No cluster file — estimating CCF from VAF × 2 / purity (diploid approximation)")
        muts["ccf"] = (muts["vaf"] * 2 / purity).clip(upper=1.0)

    # ── compute mutation copy number ──────────────────────────────────────
    muts["mut_copy_number"] = muts.apply(
        lambda r: compute_mut_copy_number(
            r["vaf"], purity, r.get("total_cn", np.nan),
            r.get("seg_cf", 1.0)
        ), axis=1
    )

    # ── classify ──────────────────────────────────────────────────────────
    muts["CLS"] = muts.apply(
        lambda r: classify_timing(r, clonal_thr, subclonal_thr), axis=1
    )

    counts = muts["CLS"].value_counts()
    info("Classification results:")
    for cls in CLS_ORDER:
        n = counts.get(cls, 0)
        pct = 100 * n / max(len(muts), 1)
        info(f"  {cls:20s}: {n:4d}  ({pct:5.1f}%)")

    return muts

# ── output TSVs ───────────────────────────────────────────────────────────────

def write_tsv_outputs(muts, seg, sample_id, prefix):
    """Write per-mutation, per-segment, and summary TSVs."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # ── per-mutation ──────────────────────────────────────────────────────
    mut_cols = [c for c in [
        "mutation_id","chrom","pos","ref","alt","filter","is_snv",
        "vaf","t_ref","t_alt","depth","ccf","mut_copy_number",
        "major_cn","minor_cn","total_cn","seg_cf","CLS","qual"
    ] if c in muts.columns]
    mut_out = muts[mut_cols].copy()
    for c in ("vaf","ccf","mut_copy_number","seg_cf","qual"):
        if c in mut_out.columns:
            mut_out[c] = mut_out[c].round(5)
    mut_path = f"{prefix}_mutations.tsv"
    mut_out.to_csv(mut_path, sep="\t", index=False)
    info(f"Written: {mut_path}  ({len(mut_out)} rows)")

    # sanity: check CLS distribution
    for cls in ["early clonal","late clonal","clonal [NA]","subclonal"]:
        n = (mut_out["CLS"] == cls).sum()
        if n == 0:
            warn(f"No mutations classified as '{cls}' — "
                 f"check purity, CN segments, and CCF thresholds")

    # ── per-segment summary ───────────────────────────────────────────────
    if not seg.empty and "CLS" in muts.columns:
        seg_rows = []
        for _, s in seg.iterrows():
            mask = (
                (muts["chrom"] == s["chrom"]) &
                (muts["pos"]   >= s["start"]) &
                (muts["pos"]   <= s["end"])
            )
            sub = muts[mask]
            cls_counts = sub["CLS"].value_counts()
            seg_rows.append(dict(
                chrom=s["chrom"], start=int(s["start"]), end=int(s["end"]),
                major_cn=s["major_cn"], minor_cn=s["minor_cn"],
                total_cn=s["total_cn"], cf=s["cf"],
                n_muts=len(sub),
                n_early_clonal=cls_counts.get("early clonal",0),
                n_late_clonal=cls_counts.get("late clonal",0),
                n_clonal_na=cls_counts.get("clonal [NA]",0),
                n_subclonal=cls_counts.get("subclonal",0),
                n_unclassified=cls_counts.get("unclassified",0),
                dominant_cls=cls_counts.index[0] if len(cls_counts) else "none",
            ))
        seg_df = pd.DataFrame(seg_rows)
        seg_path = f"{prefix}_segments.tsv"
        seg_df.to_csv(seg_path, sep="\t", index=False)
        info(f"Written: {seg_path}  ({len(seg_df)} rows)")

    # ── summary ───────────────────────────────────────────────────────────
    total = len(muts)
    cls_counts = muts["CLS"].value_counts()
    rows = []
    for cls in CLS_ORDER:
        n = cls_counts.get(cls, 0)
        rows.append(dict(
            sample_id=sample_id, CLS=cls,
            n=n, fraction=round(n/max(total,1), 5)
        ))
    rows.append(dict(sample_id=sample_id, CLS="TOTAL", n=total, fraction=1.0))
    summ = pd.DataFrame(rows)
    summ_path = f"{prefix}_summary.tsv"
    summ.to_csv(summ_path, sep="\t", index=False)
    info(f"Written: {summ_path}")

    return mut_out

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_overview(pdf, muts, sample_id, purity, clonal_thr, subclonal_thr):
    """Page 1 — text summary table."""
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f"Mutation Timing Classification  |  {sample_id}",
                 fontsize=13, fontweight="bold", y=0.97)
    ax.axis("off")

    total = len(muts)
    cls_counts = muts["CLS"].value_counts() if total > 0 else pd.Series(dtype=int)
    n_snv  = muts["is_snv"].sum()    if "is_snv" in muts.columns else total
    n_pass = (muts["filter"] == "PASS").sum() if "filter" in muts.columns else total
    n_cn   = muts["major_cn"].notna().sum()   if "major_cn" in muts.columns else 0
    n_ccf  = muts["ccf"].notna().sum()        if "ccf" in muts.columns else 0

    rows = [
        ["Sample",                    sample_id],
        ["Tumour purity",             f"{purity:.3f}  ({purity*100:.1f}%)"],
        ["Clonal threshold (CCF)",    f"≥ {clonal_thr:.2f}"],
        ["Subclonal threshold (CCF)", f"< {subclonal_thr:.2f}"],
        ["Total mutations",           str(total)],
        ["SNVs",                      str(int(n_snv))],
        ["PASS filter",               str(int(n_pass))],
        ["CN-assigned",               f"{int(n_cn)}  ({100*n_cn/max(total,1):.1f}%)"],
        ["CCF available",             f"{int(n_ccf)}  ({100*n_ccf/max(total,1):.1f}%)"],
        ["—", ""],
    ]
    for cls in CLS_ORDER:
        n = cls_counts.get(cls, 0)
        pct = 100 * n / max(total, 1)
        rows.append([cls, f"{n}  ({pct:.1f}%)"])

    tbl = ax.table(cellText=rows, colLabels=["Property","Value"],
                   loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.2, 2.0)
    for j in range(2):
        tbl[0,j].set_facecolor("#2C3E50")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        shade = "#F2F4F4" if i % 2 == 0 else "white"
        for j in range(2): tbl[i,j].set_facecolor(shade)
    # colour CLS rows
    for i, cls in enumerate(CLS_ORDER, start=11):
        if cls in CLS_COLOURS and i <= len(rows):
            tbl[i,0].set_facecolor(CLS_COLOURS[cls])
            tbl[i,0].set_text_props(color="white", fontweight="bold")

    _caption(fig,
        "Summary of the mutation timing classification. "
        "Classification follows Gerstung et al. (Nature 2020 PCAWG): "
        "early clonal = co-amplified (mut. copy number > 1); "
        "late clonal = on LOH segment (minor_cn = 0, total_cn > 2); "
        "clonal [NA] = clonal, timing unresolvable; "
        "subclonal = CCF below clonal threshold.",
        y=0.01)
    _save(pdf, fig)


def plot_cls_bar(pdf, muts, sample_id):
    """Page 2 — CLS count bar chart."""
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"Timing Class Counts  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)

    cls_present = [c for c in CLS_ORDER if c in muts["CLS"].values]
    counts      = [muts["CLS"].eq(c).sum() for c in cls_present]
    colours     = [CLS_COLOURS[c] for c in cls_present]

    bars = ax.bar(cls_present, counts, color=colours, edgecolor="white",
                  alpha=0.88, width=0.55)
    total = len(muts)
    for bar, n in zip(bars, counts):
        pct = 100 * n / max(total, 1)
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                f"{n}\n({pct:.1f}%)", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(counts or [1]) * 1.25)
    ax.tick_params(axis="x", labelsize=9, rotation=15)
    _style(ax, "Mutations per Timing Class", "Timing class", "Mutation count")
    fig.subplots_adjust(bottom=0.22, top=0.88)
    _caption(fig,
        "Number of mutations in each timing class. "
        "A high 'early clonal' fraction indicates many clonal amplification events. "
        "A dominant 'subclonal' class suggests an actively evolving tumour with "
        "multiple distinct subclones.",
        y=0.02)
    _save(pdf, fig)


def plot_cls_pie(pdf, muts, sample_id):
    """Page 3 — CLS pie chart."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.suptitle(f"Timing Class Proportions  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    ax.set_position([0.1, 0.12, 0.8, 0.72])

    cls_present = [c for c in CLS_ORDER if muts["CLS"].eq(c).sum() > 0]
    counts      = [muts["CLS"].eq(c).sum() for c in cls_present]
    colours     = [CLS_COLOURS[c] for c in cls_present]

    if sum(counts) == 0:
        ax.text(0.5, 0.5, "No classified mutations", ha="center",
                va="center", transform=ax.transAxes, fontsize=12, color="#888888")
    else:
        wedges, texts, autos = ax.pie(
            counts, labels=cls_present, colors=colours,
            autopct="%1.1f%%", startangle=90, pctdistance=0.78,
            textprops={"fontsize": 9})
        for at in autos:
            at.set_fontsize(8.5); at.set_color("white"); at.set_fontweight("bold")

    _caption(fig,
        "Proportion of mutations in each timing class. "
        "The relative balance between clonal (blue/purple/green) and subclonal (red) "
        "mutations reflects the evolutionary mode of the tumour.",
        y=0.02)
    _save(pdf, fig)


def plot_vaf_by_cls(pdf, muts, sample_id):
    """Page 4 — VAF distributions coloured by timing class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"VAF Distribution by Timing Class  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.18, top=0.88)

    bins = np.linspace(0, 1, 41)
    cls_present = [c for c in CLS_ORDER if muts["CLS"].eq(c).sum() > 0]
    for cls in cls_present:
        sub = muts[muts["CLS"] == cls]["vaf"].dropna()
        if len(sub) == 0: continue
        ax.hist(sub, bins=bins, color=CLS_COLOURS[cls], alpha=0.62,
                edgecolor="none", label=f"{cls} (n={len(sub)})")
    _cls_legend(ax, cls_present)
    _style(ax, "VAF Distribution by Timing Class",
           "Variant Allele Frequency (VAF)", "Mutation count")
    _caption(fig,
        "VAF distributions per timing class. "
        "Clonal mutations (blue/green/purple) should cluster near VAF = purity/2 "
        "on diploid segments. Subclonal mutations (red) typically peak at lower VAF. "
        "Early clonal mutations on gained alleles can appear at higher VAF.",
        y=0.02)
    _save(pdf, fig)


def plot_ccf_by_cls(pdf, muts, sample_id, clonal_thr, subclonal_thr):
    """Page 5 — CCF distributions with threshold lines."""
    if muts["ccf"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"CCF Distribution by Timing Class  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.18, top=0.88)

    bins = np.linspace(0, 1, 41)
    cls_present = [c for c in CLS_ORDER if muts["CLS"].eq(c).sum() > 0]
    for cls in cls_present:
        sub = muts[muts["CLS"] == cls]["ccf"].dropna()
        if len(sub) == 0: continue
        ax.hist(sub, bins=bins, color=CLS_COLOURS[cls], alpha=0.62,
                edgecolor="none", label=f"{cls} (n={len(sub)})")
    ax.axvline(clonal_thr,    color="#333333", ls="--", lw=1.5,
               label=f"Clonal threshold = {clonal_thr:.2f}")
    ax.axvline(subclonal_thr, color="#888888", ls=":",  lw=1.2,
               label=f"Subclonal threshold = {subclonal_thr:.2f}")
    _cls_legend(ax, cls_present)
    _style(ax, "CCF Distribution by Timing Class",
           "Cancer Cell Fraction (CCF)", "Mutation count")
    _caption(fig,
        "CCF per timing class with classification thresholds marked. "
        "Mutations above the clonal threshold (dashed) are classified as clonal; "
        "the timing sub-class (early/late/NA) is determined from copy number. "
        "Mutations below the subclonal threshold (dotted) are classified as subclonal.",
        y=0.02)
    _save(pdf, fig)


def plot_mut_copy_number(pdf, muts, sample_id):
    """Page 6 — mutation copy number distribution."""
    if "mut_copy_number" not in muts.columns or muts["mut_copy_number"].isna().all():
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Mutation Copy Number  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.35)

    # left: histogram of m coloured by CLS
    bins = np.linspace(0, 5, 41)
    cls_present = [c for c in CLS_ORDER if muts["CLS"].eq(c).sum() > 0]
    for cls in cls_present:
        sub = muts[muts["CLS"] == cls]["mut_copy_number"].dropna()
        if len(sub) == 0: continue
        ax1.hist(sub, bins=bins, color=CLS_COLOURS[cls], alpha=0.62,
                 edgecolor="none", label=cls)
    ax1.axvline(1.2, color="#333333", ls="--", lw=1.4,
                label="m = 1.2  (early clonal cutoff)")
    _cls_legend(ax1, cls_present)
    _style(ax1, "Mutation Copy Number Distribution",
           "Mutation copy number (m)", "Count")

    # right: scatter CCF vs m
    sub_all = muts.dropna(subset=["ccf","mut_copy_number"])
    for cls in cls_present:
        s = sub_all[sub_all["CLS"] == cls]
        ax2.scatter(s["mut_copy_number"], s["ccf"],
                    color=CLS_COLOURS[cls], s=10, alpha=0.45,
                    edgecolors="none", label=cls)
    ax2.axvline(1.2, color="#333333", ls="--", lw=1.0)
    ax2.set_xlim(0, 5); ax2.set_ylim(0, 1.05)
    _cls_legend(ax2, cls_present)
    _style(ax2, "CCF vs Mutation Copy Number",
           "Mutation copy number (m)", "CCF")

    _caption(fig,
        "Mutation copy number m = estimated number of allelic copies carrying the mutation. "
        "m > 1 (right of dashed line) indicates the mutation was present before a copy-number "
        "gain (early clonal). m ≈ 1 with clonal CCF is late clonal or clonal [NA]. "
        "Right: CCF vs m — early clonal mutations cluster top-right (high CCF, high m).",
        y=0.02)
    _save(pdf, fig)


def plot_cn_state(pdf, muts, sample_id):
    """Page 7 — CN state vs timing class."""
    needed = {"major_cn","minor_cn","total_cn"}
    if not needed.issubset(muts.columns) or muts["major_cn"].isna().all():
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Copy-Number State by Timing Class  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.38)

    df2 = muts.dropna(subset=["major_cn","minor_cn"]).copy()
    df2["cn_state"] = (df2["major_cn"].astype(int).astype(str) + "+" +
                       df2["minor_cn"].astype(int).astype(str))

    # left: CN state bar coloured by early/late
    top_states = df2["cn_state"].value_counts().head(8).index
    x = np.arange(len(top_states)); w = 0.18
    cls_to_plot = [c for c in CLS_ORDER if c in df2["CLS"].values]
    for i, cls in enumerate(cls_to_plot):
        counts = [df2[(df2["cn_state"]==s) & (df2["CLS"]==cls)].shape[0]
                  for s in top_states]
        axes[0].bar(x + i*w, counts, width=w,
                    color=CLS_COLOURS[cls], alpha=0.85,
                    edgecolor="white", label=cls)
    axes[0].set_xticks(x + w*len(cls_to_plot)/2)
    axes[0].set_xticklabels(top_states, rotation=35, ha="right", fontsize=8)
    axes[0].legend(fontsize=7.5, loc="upper right")
    _style(axes[0], "Timing Class per CN State  (top 8 states)",
           "CN state  (major+minor)", "Mutation count")

    # right: scatter major vs minor CN, coloured by CLS
    sub = muts.dropna(subset=["major_cn","minor_cn","CLS"]).copy()
    jitter = 0.15
    for cls in cls_to_plot:
        s = sub[sub["CLS"]==cls]
        axes[1].scatter(
            s["minor_cn"] + np.random.uniform(-jitter, jitter, len(s)),
            s["major_cn"] + np.random.uniform(-jitter, jitter, len(s)),
            color=CLS_COLOURS[cls], s=12, alpha=0.45,
            edgecolors="none", label=cls)
    axes[1].set_xlim(-0.5, sub["minor_cn"].max()+1)
    axes[1].set_ylim(-0.5, min(sub["major_cn"].max()+1, 8))
    axes[1].legend(fontsize=8, markerscale=1.5)
    _style(axes[1], "Major vs Minor CN (jittered)", "Minor CN", "Major CN")

    _caption(fig,
        "Left: timing class breakdown per copy-number state. "
        "Diploid (1+1) segments can only host clonal [NA] or subclonal mutations. "
        "High-CN states (e.g. 2+0, 2+1, 3+0) are where early/late clonal distinctions arise. "
        "Right: scatter of major vs minor CN per mutation, coloured by timing class.",
        y=0.02)
    _save(pdf, fig)


def plot_genome_landscape(pdf, muts, sample_id):
    """Page 8 — genome-wide VAF/CCF landscape coloured by CLS."""
    if muts["chrom"].isna().all():
        return

    present = [c for c in CHR_ORDER if c in muts["chrom"].dropna().values]
    if not present:
        return

    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    df2 = muts[muts["chrom"].isin(present)].copy()
    df2["gx"] = df2["chrom"].map(offsets).fillna(0) + df2.get("pos", pd.Series(dtype=float)).fillna(0)

    has_ccf = "ccf" in df2.columns and df2["ccf"].notna().any()
    nrows = 2 if has_ccf else 1

    fig, axes = plt.subplots(nrows, 1, figsize=(14, 5*nrows+1), sharex=True)
    if nrows == 1:
        axes = [axes]
    fig.suptitle(f"Genome-wide Mutation Landscape  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.08, bottom=0.12, top=0.91)

    cls_present = [c for c in CLS_ORDER if c in df2["CLS"].values]
    for cls in cls_present:
        sub = df2[df2["CLS"]==cls]
        axes[0].scatter(sub["gx"], sub["vaf"],
                        color=CLS_COLOURS[cls], s=7, alpha=0.5,
                        edgecolors="none", label=cls)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7.5, markerscale=1.5, loc="upper right", ncol=2)
    _style(axes[0], "VAF across the Genome  (coloured by timing class)",
           "", "VAF")

    if has_ccf:
        for cls in cls_present:
            sub = df2[df2["CLS"]==cls]
            axes[1].scatter(sub["gx"], sub["ccf"],
                            color=CLS_COLOURS[cls], s=7, alpha=0.5,
                            edgecolors="none", label=cls)
        axes[1].set_ylim(0, 1.05)
        _style(axes[1], "CCF across the Genome  (coloured by timing class)",
               "Genomic position (hg38)", "CCF")

    for ch in present:
        for ax in axes:
            ax.axvline(offsets[ch], color="#EEEEEE", lw=0.6, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch, 100e6)/2
        axes[-1].text(mid, -0.07, ch.replace("chr",""),
                      ha="center", fontsize=6, color="#555555",
                      transform=axes[-1].get_xaxis_transform())
    axes[0].set_xlim(0, cum)

    _caption(fig,
        "Genome-wide scatter of all mutations coloured by timing class. "
        "Top: VAF; bottom (if CCF available): CCF. "
        "Early clonal mutations (blue) at elevated VAF indicate CN gain regions. "
        "Subclonal mutations (red) cluster at low VAF/CCF across all chromosomes.",
        y=0.02)
    _save(pdf, fig)


def plot_chrom_cls(pdf, muts, sample_id):
    """Page 9 — per-chromosome CLS composition (100% stacked)."""
    if muts["chrom"].isna().all():
        return

    present     = [c for c in CHR_ORDER if c in muts["chrom"].dropna().values]
    other       = sorted(set(muts["chrom"].dropna().unique()) - set(CHR_ORDER))
    chrom_order = present + other

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(f"Chromosomal Distribution by Timing Class  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.55, bottom=0.12, top=0.91)

    x      = np.arange(len(chrom_order))
    totals = muts.groupby("chrom").size().reindex(chrom_order, fill_value=0).values.astype(float)

    # absolute
    bottom = np.zeros(len(chrom_order))
    for cls in CLS_ORDER:
        counts = (muts[muts["CLS"]==cls]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        ax1.bar(x, counts, bottom=bottom, color=CLS_COLOURS[cls],
                edgecolor="white", linewidth=0.3, label=cls)
        bottom += counts
    ax1.set_xticks(x)
    ax1.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7)
    ax1.legend(fontsize=8, loc="upper right")
    _style(ax1, "Mutations per Chromosome  (stacked by timing class)",
           "Chromosome", "Count")

    # 100%
    bottom = np.zeros(len(chrom_order))
    for cls in CLS_ORDER:
        counts = (muts[muts["CLS"]==cls]
                  .groupby("chrom").size()
                  .reindex(chrom_order, fill_value=0).values.astype(float))
        fracs  = np.where(totals > 0, counts/totals, 0)
        ax2.bar(x, fracs, bottom=bottom, color=CLS_COLOURS[cls],
                edgecolor="white", linewidth=0.3, label=cls)
        bottom += fracs
    ax2.set_xticks(x)
    ax2.set_xticklabels(chrom_order, rotation=45, ha="right", fontsize=7)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8, loc="upper right")
    _style(ax2, "Timing Class Composition per Chromosome  (100% stacked)",
           "Chromosome", "Fraction")

    _caption(fig,
        "Top: absolute mutation counts per chromosome, coloured by timing class. "
        "Bottom: fractional composition. Chromosomes with a high 'early clonal' fraction "
        "are likely carrying gains that predate the clonal expansion.",
        y=0.02)
    _save(pdf, fig)


def plot_cn_segment_timing(pdf, muts, seg, sample_id):
    """Page 10 — per-segment view: dominant CLS + CN state."""
    needed = {"major_cn","minor_cn","total_cn"}
    if not needed.issubset(muts.columns) or muts["major_cn"].isna().all():
        return
    if seg.empty:
        return

    present = [c for c in CHR_ORDER if c in muts["chrom"].dropna().values]
    if not present:
        return

    offsets, cum = {}, 0
    for ch in present:
        offsets[ch] = cum; cum += CHR_LENGTHS.get(ch, 100e6)

    # per-segment dominant CLS
    seg_plot = []
    for _, s in seg.iterrows():
        if s["chrom"] not in present: continue
        mask = ((muts["chrom"] == s["chrom"]) &
                (muts["pos"]   >= s["start"]) &
                (muts["pos"]   <= s["end"]))
        sub = muts[mask]
        if len(sub) == 0: continue
        dominant = sub["CLS"].value_counts().index[0]
        gx_start = offsets[s["chrom"]] + s["start"]
        gx_end   = offsets[s["chrom"]] + s["end"]
        seg_plot.append(dict(
            gx_start=gx_start, gx_end=gx_end,
            total_cn=s["total_cn"], major_cn=s["major_cn"],
            minor_cn=s["minor_cn"],
            dominant=dominant, n_muts=len(sub)
        ))

    if not seg_plot:
        return

    sp = pd.DataFrame(seg_plot)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Per-segment Timing & CN State  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.subplots_adjust(hspace=0.10, bottom=0.12, top=0.91)

    for _, row in sp.iterrows():
        col = CLS_COLOURS.get(row["dominant"], "#AAAAAA")
        ax1.barh(0, row["gx_end"]-row["gx_start"],
                 left=row["gx_start"], height=0.8,
                 color=col, alpha=0.75, edgecolor="none")

    ax1.set_ylim(-0.5, 0.5); ax1.set_yticks([])
    ax1.set_title("Dominant timing class per CN segment",
                  fontsize=10, fontweight="bold", pad=6)
    patches = [mpatches.Patch(color=CLS_COLOURS[c], label=c, alpha=0.8)
               for c in CLS_ORDER if c in sp["dominant"].values]
    ax1.legend(handles=patches, fontsize=8, loc="upper right")

    # total CN track
    for _, row in sp.iterrows():
        ax2.barh(0, row["gx_end"]-row["gx_start"],
                 left=row["gx_start"],
                 height=row["total_cn"]/8 * 0.7 + 0.1,
                 color="#4C72B0", alpha=0.5, edgecolor="none")

    ax2.set_ylim(-0.1, 1.1); ax2.set_yticks([])
    ax2.set_title("Total copy number per segment  (bar height ∝ CN)",
                  fontsize=10, fontweight="bold", pad=6)

    for ch in present:
        for ax in (ax1, ax2):
            ax.axvline(offsets[ch], color="#DDDDDD", lw=0.7, zorder=0)
        mid = offsets[ch] + CHR_LENGTHS.get(ch,100e6)/2
        ax2.text(mid, -0.08, ch.replace("chr",""),
                 ha="center", fontsize=6, color="#555555",
                 transform=ax2.get_xaxis_transform())
    ax1.set_xlim(0, cum)

    _caption(fig,
        "Top: genome-wide track showing the dominant timing class of mutations within each "
        "CN segment. Blue = early clonal, purple = late clonal, green = clonal [NA], red = subclonal. "
        "Bottom: total copy number per segment (bar height proportional to CN). "
        "Segments with high CN and 'early clonal' colour predate the copy-number event.",
        y=0.02)
    _save(pdf, fig)


def plot_vaf_ccf_scatter(pdf, muts, purity, sample_id):
    """Page 11 — VAF vs CCF, showing CN correction effect."""
    if muts["ccf"].isna().all():
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"VAF vs CCF  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.18, top=0.88, wspace=0.35)

    cls_present = [c for c in CLS_ORDER if c in muts["CLS"].values]
    sub = muts.dropna(subset=["vaf","ccf"])

    for cls in cls_present:
        s = sub[sub["CLS"]==cls]
        ax1.scatter(s["vaf"], s["ccf"],
                    color=CLS_COLOURS[cls], s=10, alpha=0.45,
                    edgecolors="none", label=cls)
    xr = np.linspace(0, 0.6, 100)
    ax1.plot(xr, 2*xr, color="#AAAAAA", ls="--", lw=1.3,
             label="CCF = 2×VAF  (diploid)")
    ax1.set_xlim(0, 1.0); ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=7.5)
    _style(ax1, "VAF vs CCF  (per mutation)", "VAF", "CCF")

    # right: VAF coloured by mut_copy_number
    if "mut_copy_number" in muts.columns and muts["mut_copy_number"].notna().any():
        sub2 = muts.dropna(subset=["vaf","ccf","mut_copy_number"])
        sc   = ax2.scatter(sub2["vaf"], sub2["ccf"],
                           c=sub2["mut_copy_number"].clip(0,5),
                           cmap="plasma", s=10, alpha=0.55,
                           vmin=0, vmax=4, edgecolors="none")
        cb   = fig.colorbar(sc, ax=ax2, shrink=0.7)
        cb.set_label("Mutation copy number (m)", fontsize=8)
        ax2.set_xlim(0, 1.0); ax2.set_ylim(0, 1.1)
        ax2.plot(xr, 2*xr, color="#AAAAAA", ls="--", lw=1.3)
        _style(ax2, "VAF vs CCF  (coloured by m)",
               "VAF", "CCF")

    _caption(fig,
        "VAF vs CCF per mutation. Points above the dashed line (CCF = 2×VAF) have "
        "been CN-corrected upward — they sit on amplified segments. "
        "Right panel: colour = mutation copy number m; high-m points (yellow/bright) "
        "are early clonal mutations on gained alleles.",
        y=0.02)
    _save(pdf, fig)


def plot_timing_waterfall(pdf, muts, sample_id):
    """Page 12 — waterfall-style ranked CCF coloured by CLS."""
    if muts["ccf"].isna().all():
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f"Ranked CCF Waterfall  |  {sample_id}",
                 fontsize=12, fontweight="bold", y=0.97)
    fig.subplots_adjust(bottom=0.18, top=0.88)

    plot_df = muts[["ccf","CLS"]].dropna(subset=["ccf"]).sort_values("ccf", ascending=False)
    x = np.arange(len(plot_df))
    cols = [CLS_COLOURS.get(c, "#AAAAAA") for c in plot_df["CLS"]]
    ax.bar(x, plot_df["ccf"].values, color=cols, edgecolor="none", width=1.0)

    cls_present = [c for c in CLS_ORDER if c in plot_df["CLS"].values]
    patches = [mpatches.Patch(color=CLS_COLOURS[c], label=c) for c in cls_present]
    ax.legend(handles=patches, fontsize=8.5, loc="upper right")
    ax.set_xlim(0, len(plot_df))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Mutations  (ranked by CCF, high→low)", fontsize=9)
    ax.set_ylabel("Cancer Cell Fraction (CCF)", fontsize=9)
    ax.set_title("Ranked CCF — Waterfall Plot", fontsize=11,
                 fontweight="bold", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _caption(fig,
        "Each bar = one mutation, ranked by CCF from highest (left) to lowest (right), "
        "coloured by timing class. "
        "The colour pattern from left to right reveals the clonal architecture: "
        "early and late clonal mutations on the left, subclonal on the right.",
        y=0.02)
    _save(pdf, fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Somatic mutation timing classification (Gerstung 2020 logic).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
        Output files (written before plots):
          <prefix>_mutations.tsv   — per-mutation timing + CN + CCF
          <prefix>_segments.tsv    — per-segment summary
          <prefix>_summary.tsv     — CLS counts and fractions
          <prefix>_timing_report.pdf — visual report
        """
    )
    parser.add_argument("--vcf",      "-v", required=True,
                        help="DeepSomatic VCF (*.vcf.gz or *.vcf)")
    parser.add_argument("--seg",      "-s", required=True,
                        help="FACETS CN segment TSV (nA/nB or tcn/lcn columns)")
    parser.add_argument("--purity",   "-p", required=True,
                        help="Purity text file (single float, e.g. 0.83)")
    parser.add_argument("--clusters", "-c", default=None,
                        help="PyClone6 results.tsv or VIBER mutations.tsv "
                             "(provides CCF per mutation; optional)")
    parser.add_argument("--sample",   default="sample",
                        help="Sample ID used in plot titles and output column")
    parser.add_argument("--prefix",   "-o", default=None,
                        help="Output path prefix (default: <sample>_muttime/<sample>)")
    parser.add_argument("--clonal-threshold",    type=float, default=0.80,
                        help="CCF ≥ this → clonal  (default: 0.80)")
    parser.add_argument("--subclonal-threshold", type=float, default=0.20,
                        help="CCF < this → subclonal  (default: 0.20). "
                             "Mutations between the two thresholds are also subclonal.")
    parser.add_argument("--pass-only", action="store_true", default=False,
                        help="Restrict to PASS variants only (default: use all)")
    args = parser.parse_args()

    import textwrap   # already imported at top but guard here for --help

    sid = args.sample
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = f"{sid}_muttime/{sid}"

    print(f"[muttime] Sample: {sid}")
    print(f"[muttime] Clonal threshold:    CCF ≥ {args.clonal_threshold}")
    print(f"[muttime] Subclonal threshold: CCF < {args.subclonal_threshold}")

    # ── load inputs ───────────────────────────────────────────────────────
    purity   = load_purity(args.purity)
    muts     = load_vcf(args.vcf)
    seg      = load_segments(args.seg)
    clusters = load_clusters(args.clusters)

    if muts.empty:
        warn("VCF is empty — writing empty output files and a warning-only PDF")

    if args.pass_only and not muts.empty:
        n_before = len(muts)
        muts = muts[muts["filter"] == "PASS"].copy()
        info(f"--pass-only: kept {len(muts)}/{n_before} PASS variants")

    # ── assign CN segments ────────────────────────────────────────────────
    if not muts.empty:
        muts = assign_segments(muts, seg)

    # ── classify ──────────────────────────────────────────────────────────
    if not muts.empty:
        muts = run_classification(
            muts, purity, clusters,
            args.clonal_threshold, args.subclonal_threshold)
    else:
        for col in ("ccf","mut_copy_number","CLS"):
            muts[col] = pd.NA

    # ── write TSVs first ──────────────────────────────────────────────────
    print(f"[muttime] Writing output TSVs to: {prefix}_*")
    mut_out = write_tsv_outputs(muts, seg, sid, prefix)

    # ── generate PDF ──────────────────────────────────────────────────────
    pdf_path = f"{prefix}_timing_report.pdf"
    print(f"[muttime] Writing report: {pdf_path}")

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "figure.facecolor": "white",
        "axes.facecolor":   "#FAFAFA",
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.5,
    })

    with PdfPages(pdf_path) as pdf:
        d = pdf.infodict()
        d["Title"]  = f"Mutation Timing Report — {sid}"
        d["Author"] = "muttime_classify.py"

        plot_overview(pdf, muts, sid, purity,
                      args.clonal_threshold, args.subclonal_threshold)
        plot_cls_bar(pdf, muts, sid)
        plot_cls_pie(pdf, muts, sid)
        plot_vaf_by_cls(pdf, muts, sid)
        plot_ccf_by_cls(pdf, muts, sid,
                        args.clonal_threshold, args.subclonal_threshold)
        plot_mut_copy_number(pdf, muts, sid)
        plot_cn_state(pdf, muts, sid)
        plot_genome_landscape(pdf, muts, sid)
        plot_chrom_cls(pdf, muts, sid)
        plot_cn_segment_timing(pdf, muts, seg, sid)
        plot_vaf_ccf_scatter(pdf, muts, purity, sid)
        plot_timing_waterfall(pdf, muts, sid)

    print(f"[muttime] Done.")
    print(f"[muttime] TSVs:   {prefix}_mutations.tsv")
    print(f"[muttime]         {prefix}_segments.tsv")
    print(f"[muttime]         {prefix}_summary.tsv")
    print(f"[muttime] Report: {pdf_path}")


if __name__ == "__main__":
    import textwrap
    main()