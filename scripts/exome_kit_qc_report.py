#!/usr/bin/env python3
"""
sample_capture_qc_report.py

Single-sample capture/exome QC report built entirely from files a
standard WES Snakemake pipeline already produces (or one extra common
command away from producing). No truth set / ground truth required --
every check here is intrinsic to the sample's own data. Run it once per
sample/capture, then compare two PDFs by eye (e.g. the same sample
captured with a commercial panel vs. a homegrown panel).

Inputs (all optional except --reference and --out; missing inputs just
skip the pages that need them and are listed on the title page so you
know what to generate manually):

  --reference FASTA          reference genome (indexed .fai alongside,
                              or one will be built)
  --target-bed BED           capture target BED (informational; GC-bias
                              plotting uses the mosdepth regions file's
                              own coordinates, not this)
  --mosdepth-prefix PREFIX   expects PREFIX.mosdepth.summary.txt,
                              PREFIX.mosdepth.global.dist.txt, and
                              (if mosdepth was run with --by target.bed)
                              PREFIX.regions.bed.gz
                              -> generate with:
                              mosdepth --by target.bed -t 4 PREFIX sample.bam
  --samtools-stats FILE      `samtools stats sample.bam > FILE`
  --vcf FILE                 variant calls (.vcf or .vcf.gz), somatic or
                              germline -- intrinsic QC only, no truth
                              comparison
  --facets-cncf FILE         FACETS per-segment cncf output table
  --facets-summary FILE      FACETS purity/ploidy/dipLogR summary
                              (2-col key/value TSV, or 1-row CSV with
                              columns purity/ploidy/dipLogR)
  --cnaqc-table FILE         any CNAqc-exported CSV/TSV table (rendered
                              generically -- see parse_generic_table)

Example (typical exome_wes rule):

    python sample_capture_qc_report.py \\
        --sample-name TUMOR01_homegrown \\
        --reference /ref/GRCh38.fa \\
        --target-bed panel/homegrown_targets.bed \\
        --mosdepth-prefix qc/TUMOR01_homegrown \\
        --samtools-stats qc/TUMOR01_homegrown.samtools_stats.txt \\
        --vcf calls/TUMOR01_homegrown.deepsomatic.vcf.gz \\
        --facets-cncf cnv/TUMOR01_homegrown_cncf.txt \\
        --facets-summary cnv/TUMOR01_homegrown_summary.txt \\
        --out qc_reports/TUMOR01_homegrown.pdf

No Snakemake-specific bindings -- callable directly from a `shell:` rule.
"""

import argparse
import gzip
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

C_MAIN = "#2C6E9B"
C_ACCENT = "#D96C2C"
C_OK = "#4E9C4E"
C_WARN = "#C9A227"
C_BAD = "#B23A48"
C_GREY = "#888888"

plt.rcParams.update({
    "figure.figsize": (11.69, 8.27),  # A4 landscape
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
A4_LANDSCAPE = (11.69, 8.27)

DEFAULT_CHROM_REGEX = r"^(chr)?([0-9]{1,2}|X|Y|M|MT)$"


# --------------------------------------------------------------------------
# FASTA / .fai handling (pure python, no samtools/pysam dependency)
# --------------------------------------------------------------------------

def read_or_build_fai(fasta_path):
    """Return dict: chrom -> (length, byte_offset, linebases, linewidth).
    Reads an existing .fai if present; otherwise builds one in memory
    (and tries to write it alongside the FASTA for reuse)."""
    fasta_path = Path(fasta_path)
    fai_path = fasta_path.with_suffix(fasta_path.suffix + ".fai")

    if fai_path.exists():
        fai = {}
        with open(fai_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                name, length, offset, linebases, linewidth = parts[:5]
                fai[name] = (int(length), int(offset), int(linebases), int(linewidth))
        return fai

    print(f"  (no .fai found for {fasta_path}, building index -- this may take a moment)",
          file=sys.stderr)
    fai = {}
    with open(fasta_path, "rb") as fh:
        name = None
        length = 0
        offset = 0
        linebases = None
        linewidth = None
        pos = 0
        while True:
            line = fh.readline()
            if not line:
                if name is not None:
                    fai[name] = (length, offset, linebases, linewidth)
                break
            pos_after = fh.tell()
            line_len = pos_after - pos
            if line.startswith(b">"):
                if name is not None:
                    fai[name] = (length, offset, linebases, linewidth)
                name = line[1:].split()[0].decode()
                length = 0
                linebases = None
                linewidth = None
                offset = pos_after
            else:
                seq_len = len(line.rstrip(b"\n"))
                if linebases is None:
                    linebases = seq_len
                    linewidth = line_len
                length += seq_len
            pos = pos_after

    try:
        with open(fai_path, "w") as out:
            for name, (length, offset, linebases, linewidth) in fai.items():
                out.write(f"{name}\t{length}\t{offset}\t{linebases}\t{linewidth}\n")
    except OSError:
        pass  # read-only filesystem etc. -- fine, we just keep it in memory

    return fai


def fetch_seq(fasta_path, fai, chrom, start, end):
    """0-based half-open [start, end) fetch, tolerant of chr-prefix mismatch."""
    if chrom not in fai:
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt in fai:
            chrom = alt
        else:
            return ""
    length, offset, linebases, linewidth = fai[chrom]
    start = max(0, start)
    end = min(length, end)
    if end <= start:
        return ""
    with open(fasta_path, "rb") as fh:
        start_line = start // linebases
        start_line_pos = start % linebases
        file_start = offset + start_line * linewidth + start_line_pos
        fh.seek(file_start)
        n_bases = end - start
        n_bytes = n_bases + (n_bases // linebases) + 2
        raw = fh.read(n_bytes)
        seq = raw.replace(b"\n", b"").replace(b"\r", b"")[:n_bases]
    return seq.decode().upper()


def gc_fraction(seq):
    if not seq:
        return np.nan
    gc = seq.count("G") + seq.count("C")
    at = seq.count("A") + seq.count("T")
    total = gc + at
    return gc / total if total else np.nan


def build_genome_axis(fai, chrom_regex=DEFAULT_CHROM_REGEX):
    """Order primary chromosomes as they appear in the .fai, compute
    cumulative genomic offsets for a whole-genome x-axis."""
    pattern = re.compile(chrom_regex)
    chroms = [(name, info[0]) for name, info in fai.items() if pattern.match(name)]

    def sort_key(item):
        name, _ = item
        n = name[3:] if name.lower().startswith("chr") else name
        if n.isdigit():
            return (0, int(n))
        order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
        return (1, order.get(n.upper(), 99))

    chroms.sort(key=sort_key)
    offsets = {}
    cum = 0
    for name, length in chroms:
        offsets[name] = cum
        cum += length
    return chroms, offsets, cum


def genome_x(offsets, chrom, pos):
    if chrom not in offsets:
        alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
        if alt not in offsets:
            return np.nan
        chrom = alt
    return offsets[chrom] + pos


# --------------------------------------------------------------------------
# Parsers -- mosdepth
# --------------------------------------------------------------------------

def _open_maybe_gz(path):
    path = str(path)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def parse_mosdepth_summary(prefix):
    path = Path(f"{prefix}.mosdepth.summary.txt")
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def parse_mosdepth_global_dist(prefix):
    path = Path(f"{prefix}.mosdepth.global.dist.txt")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t", header=None,
                      names=["chrom", "depth", "proportion"])
    return df[df["chrom"] == "total"].sort_values("depth")


def parse_mosdepth_regions(prefix):
    for suffix in (".regions.bed.gz", ".regions.bed"):
        path = Path(f"{prefix}{suffix}")
        if path.exists():
            with _open_maybe_gz(path) as fh:
                df = pd.read_csv(fh, sep="\t", header=None,
                                  names=["chrom", "start", "end", "mean_depth"])
            return df
    raise FileNotFoundError(f"{prefix}.regions.bed(.gz) -- was mosdepth run with --by target.bed?")


# --------------------------------------------------------------------------
# Parsers -- samtools stats
# --------------------------------------------------------------------------

def parse_samtools_stats(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    sn = {}
    is_rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("SN\t"):
                parts = line.rstrip("\n").split("\t")
                key = parts[1].rstrip(":")
                val = parts[2].split("#")[0].strip()
                sn[key] = val
            elif line.startswith("IS\t"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    is_rows.append((int(parts[1]), int(parts[2])))
    is_df = pd.DataFrame(is_rows, columns=["insert_size", "pairs_total"])
    return sn, is_df


def sn_float(sn, key, default=np.nan):
    v = sn.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Parsers -- VCF (intrinsic QC only, no truth needed)
# --------------------------------------------------------------------------

INFO_ANNOTATIONS_OF_INTEREST = ["MQ", "FS", "SOR", "BaseQRankSum", "ReadPosRankSum", "QD"]


def parse_vcf_intrinsic(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    with _open_maybe_gz(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos, _id, ref, alt_field, qual, filt, info = f[:8]
            fmt = f[8] if len(f) > 8 else ""
            sample = f[9] if len(f) > 9 else ""

            info_dict = {}
            for kv in info.split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    info_dict[k] = v

            fmt_keys = fmt.split(":")
            sample_vals = sample.split(":")
            fmt_dict = dict(zip(fmt_keys, sample_vals))

            vaf = np.nan
            if "AF" in fmt_dict:
                try:
                    vaf = float(fmt_dict["AF"].split(",")[0])
                except ValueError:
                    pass
            elif "AD" in fmt_dict:
                try:
                    ads = [int(x) for x in fmt_dict["AD"].split(",")]
                    if len(ads) >= 2 and sum(ads) > 0:
                        vaf = ads[1] / sum(ads)
                except ValueError:
                    pass
            elif "AF" in info_dict:
                try:
                    vaf = float(info_dict["AF"].split(",")[0])
                except ValueError:
                    pass

            dp = np.nan
            if "DP" in fmt_dict:
                try:
                    dp = float(fmt_dict["DP"])
                except ValueError:
                    pass
            elif "DP" in info_dict:
                try:
                    dp = float(info_dict["DP"])
                except ValueError:
                    pass

            for alt in alt_field.split(","):
                if alt in (".", "<NON_REF>"):
                    continue
                if len(ref) == 1 and len(alt) == 1:
                    vtype = "SNV"
                    indel_len = 0
                elif len(ref) == len(alt):
                    vtype = "MNV"
                    indel_len = 0
                else:
                    vtype = "INDEL"
                    indel_len = len(alt) - len(ref)

                row = {
                    "chrom": chrom, "pos": int(pos), "ref": ref, "alt": alt,
                    "qual": float(qual) if qual not in (".", "") else np.nan,
                    "filter": filt, "type": vtype, "indel_len": indel_len,
                    "vaf": vaf, "dp": dp,
                }
                for key in INFO_ANNOTATIONS_OF_INTEREST:
                    if key in info_dict:
                        try:
                            row[key] = float(info_dict[key])
                        except ValueError:
                            pass
                rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Parsers -- FACETS
# --------------------------------------------------------------------------

def _find_col(columns, must_contain, avoid=()):
    for c in columns:
        cl = c.lower()
        if all(m in cl for m in must_contain) and not any(a in cl for a in avoid):
            return c
    return None


def parse_facets_cncf(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=None, engine="python")
    cols = list(df.columns)
    mapping = {
        "chrom": _find_col(cols, ["chr"]),
        "start": _find_col(cols, ["start"]),
        "end": _find_col(cols, ["end"]),
        "cnlr": _find_col(cols, ["cnlr", "median"], avoid=["clust"])
                or _find_col(cols, ["cnlr"], avoid=["clust"]),
        "mafr": _find_col(cols, ["mafr"], avoid=["clust"]),
        "tcn": _find_col(cols, ["tcn"]),
        "num_mark": _find_col(cols, ["num", "mark"]),
    }
    missing_required = [k for k in ("chrom", "start", "end", "cnlr") if mapping[k] is None]
    if missing_required:
        raise ValueError(
            f"Could not find columns {missing_required} in FACETS cncf file "
            f"{path}. Columns present: {cols}. You may need to adapt "
            "parse_facets_cncf() to your exact FACETS export format."
        )
    out = pd.DataFrame({k: df[v] for k, v in mapping.items() if v is not None})
    return out


def parse_facets_summary(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text()
    result = {}
    # try 1-row CSV/TSV with named columns first
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if len(df) >= 1:
            for want in ("purity", "ploidy", "diplogr", "dipLogR"):
                col = _find_col(df.columns, [want.lower()])
                if col:
                    result[want.lower().replace("diplogr", "dipLogR")] = df[col].iloc[0]
        if result:
            return result
    except Exception:
        pass
    # fall back to key=value / key\tvalue lines
    for line in text.splitlines():
        for sep in ("\t", "=", ":"):
            if sep in line:
                k, v = line.split(sep, 1)
                k = k.strip().lower()
                try:
                    result[k] = float(v.strip())
                except ValueError:
                    pass
                break
    return result


def parse_generic_table(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=None, engine="python")


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------

class Inputs:
    def __init__(self, args):
        self.sample_name = args.sample_name
        self.warnings = []
        self.notes = []

        self.fai = None
        if args.reference:
            try:
                self.fai = read_or_build_fai(args.reference)
                self.reference = args.reference
            except Exception as e:
                self.warnings.append(f"reference: {e}")
                self.reference = None
        else:
            self.reference = None
            self.warnings.append("no --reference given: GC-bias page skipped")

        self.target_bed = args.target_bed

        self.mosdepth_summary = None
        self.mosdepth_dist = None
        self.mosdepth_regions = None
        if args.mosdepth_prefix:
            try:
                self.mosdepth_summary = parse_mosdepth_summary(args.mosdepth_prefix)
            except Exception as e:
                self.warnings.append(f"mosdepth summary: {e}")
            try:
                self.mosdepth_dist = parse_mosdepth_global_dist(args.mosdepth_prefix)
            except Exception as e:
                self.warnings.append(f"mosdepth global dist: {e}")
            try:
                self.mosdepth_regions = parse_mosdepth_regions(args.mosdepth_prefix)
            except Exception as e:
                self.warnings.append(f"mosdepth regions: {e} "
                                      f"(re-run: mosdepth --by target.bed -t 4 {args.mosdepth_prefix} sample.bam)")
        else:
            self.warnings.append("no --mosdepth-prefix given: coverage pages skipped")

        self.sn = None
        self.is_df = None
        if args.samtools_stats:
            try:
                self.sn, self.is_df = parse_samtools_stats(args.samtools_stats)
            except Exception as e:
                self.warnings.append(f"samtools stats: {e}")

        self.vcf_df = None
        if args.vcf:
            try:
                self.vcf_df = parse_vcf_intrinsic(args.vcf)
                if len(self.vcf_df) == 0:
                    self.notes.append("VCF parsed but contains 0 variants")
            except Exception as e:
                self.warnings.append(f"vcf: {e}")

        self.facets_cncf = None
        if args.facets_cncf:
            try:
                self.facets_cncf = parse_facets_cncf(args.facets_cncf)
            except Exception as e:
                self.warnings.append(f"facets-cncf: {e}")

        self.facets_summary = None
        if args.facets_summary:
            try:
                self.facets_summary = parse_facets_summary(args.facets_summary)
            except Exception as e:
                self.warnings.append(f"facets-summary: {e}")

        self.cnaqc_table = None
        if args.cnaqc_table:
            try:
                self.cnaqc_table = parse_generic_table(args.cnaqc_table)
            except Exception as e:
                self.warnings.append(f"cnaqc-table: {e}")


# ---- Page 1: title / inputs -----------------------------------------------

def page_title(pdf, inp, args):
    fig = plt.figure(figsize=A4_LANDSCAPE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.5, 0.85, "Sample Capture QC Report", ha="center", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.79, inp.sample_name, ha="center", fontsize=14, color=C_MAIN)
    ax.text(0.5, 0.74, "Intrinsic QC only -- no truth set used. Compare this PDF "
                        "visually against another run of the same sample.",
             ha="center", fontsize=9.5, color="#555555")

    y = 0.63
    ax.text(0.08, y, "Inputs found:", fontsize=11, fontweight="bold")
    y -= 0.045
    found = []
    if inp.reference: found.append(f"Reference: {args.reference}")
    if args.target_bed: found.append(f"Target BED: {args.target_bed}")
    if inp.mosdepth_summary is not None: found.append("mosdepth summary")
    if inp.mosdepth_dist is not None: found.append("mosdepth global coverage distribution")
    if inp.mosdepth_regions is not None: found.append("mosdepth per-target regions")
    if inp.sn is not None: found.append("samtools stats")
    if inp.vcf_df is not None: found.append(f"VCF ({len(inp.vcf_df)} variant records)")
    if inp.facets_cncf is not None: found.append("FACETS cncf segments")
    if inp.facets_summary is not None: found.append("FACETS purity/ploidy summary")
    if inp.cnaqc_table is not None: found.append("CNAqc table")
    for item in found:
        ax.text(0.10, y, f"\u2713  {item}", fontsize=9.5, color=C_OK)
        y -= 0.038

    y -= 0.02
    ax.text(0.08, y, "Missing / skipped (generate manually if you want these pages):",
             fontsize=11, fontweight="bold")
    y -= 0.045
    if not inp.warnings:
        ax.text(0.10, y, "(none -- all optional inputs provided)", fontsize=9.5, color=C_GREY)
    for w in inp.warnings:
        ax.text(0.10, y, f"\u26a0  {w}", fontsize=8.5, color=C_BAD)
        y -= 0.036
    for n in inp.notes:
        ax.text(0.10, y, f"i  {n}", fontsize=8.5, color=C_WARN)
        y -= 0.036

    ax.text(0.08, 0.05,
             "Generated by sample_capture_qc_report.py. All checks below are "
             "heuristic pattern-detection on this sample's own data -- flags "
             "are prompts to investigate, not definitive diagnoses.",
             fontsize=7.5, color=C_GREY)
    pdf.savefig(fig)
    plt.close(fig)


# ---- Page: alignment / library QC ------------------------------------------

def page_alignment_qc(pdf, inp):
    if inp.sn is None:
        return
    sn = inp.sn
    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE)
    fig.suptitle("Alignment & Library QC (samtools stats)", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.axis("off")
    total = sn_float(sn, "raw total sequences")
    mapped = sn_float(sn, "reads mapped")
    dup = sn_float(sn, "reads duplicated")
    paired_proper = sn_float(sn, "reads properly paired")
    mismatch = sn_float(sn, "mismatches")
    bases_mapped_cig = sn_float(sn, "bases mapped (cigar)")
    err_rate = sn_float(sn, "error rate")
    avg_len = sn_float(sn, "average length")
    avg_qual = sn_float(sn, "average quality")
    insert_mean = sn_float(sn, "insert size average")
    insert_sd = sn_float(sn, "insert size standard deviation")

    rows = [
        ("Total sequences", f"{total:,.0f}" if pd.notna(total) else "--"),
        ("Reads mapped", f"{mapped:,.0f} ({mapped/total:.1%})" if pd.notna(mapped) and total else "--"),
        ("Reads properly paired", f"{paired_proper:,.0f} ({paired_proper/total:.1%})" if pd.notna(paired_proper) and total else "--"),
        ("Reads duplicated", f"{dup:,.0f} ({dup/total:.1%})" if pd.notna(dup) and total else "--"),
        ("Error rate (mismatches/bases mapped)",
         f"{err_rate:.4%}" if pd.notna(err_rate) else
         (f"{mismatch/bases_mapped_cig:.4%}" if pd.notna(mismatch) and bases_mapped_cig else "--")),
        ("Average read length", f"{avg_len:.0f} bp" if pd.notna(avg_len) else "--"),
        ("Average base quality", f"{avg_qual:.1f}" if pd.notna(avg_qual) else "--"),
        ("Insert size (mean \u00b1 sd)",
         f"{insert_mean:.0f} \u00b1 {insert_sd:.0f} bp" if pd.notna(insert_mean) else "--"),
    ]
    txt = "\n".join(f"{k}:  {v}" for k, v in rows)
    ax.text(0.02, 0.95, txt, fontsize=10.5, va="top", family="monospace")

    ax2 = axes[1]
    if inp.is_df is not None and len(inp.is_df):
        d = inp.is_df.copy()
        d = d[d["insert_size"] <= max(1000, d["insert_size"].quantile(0.99))]
        ax2.bar(d["insert_size"], d["pairs_total"], width=1.0, color=C_MAIN)
        ax2.set_xlabel("Insert size (bp)")
        ax2.set_ylabel("Read pairs")
        ax2.set_title("Insert size distribution", fontsize=10)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No insert-size histogram in samtools stats output",
                  ha="center", va="center", color=C_GREY)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ---- Page: coverage summary & uniformity -----------------------------------

def page_coverage(pdf, inp):
    if inp.mosdepth_summary is None and inp.mosdepth_dist is None and inp.mosdepth_regions is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=A4_LANDSCAPE)
    fig.suptitle("Coverage Summary & Uniformity (mosdepth)", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    if inp.mosdepth_dist is not None:
        d = inp.mosdepth_dist
        ax1.plot(d["depth"], d["proportion"], color=C_MAIN, linewidth=1.8)
        ax1.set_xlabel("Depth (x)")
        ax1.set_ylabel("Fraction of bases \u2265 depth")
        ax1.set_title("Cumulative coverage distribution", fontsize=10)
        ax1.set_ylim(0, 1.02)
        max_depth = d.loc[d["proportion"] > 0.01, "depth"].max()
        ax1.set_xlim(0, max(50, max_depth * 1.05) if pd.notna(max_depth) else 200)
        for x_ref in (10, 20, 30, 50):
            row = d[d["depth"] == x_ref]
            if len(row):
                y_ref = row["proportion"].iloc[0]
                ax1.annotate(f"{x_ref}x: {y_ref:.0%}", (x_ref, y_ref),
                              fontsize=7.5, color=C_ACCENT,
                              xytext=(5, 8), textcoords="offset points")
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No mosdepth global.dist.txt provided", ha="center", color=C_GREY)

    ax2 = axes[1]
    cv_text = None
    if inp.mosdepth_regions is not None:
        r = inp.mosdepth_regions
        mean_all = r["mean_depth"].mean()
        norm = r["mean_depth"] / mean_all if mean_all else r["mean_depth"]
        ax2.hist(norm, bins=60, range=(0, 3), color=C_MAIN, alpha=0.85)
        ax2.axvline(1.0, color="#999999", linestyle="--", linewidth=1)
        ax2.set_xlabel("Per-target normalized coverage")
        ax2.set_ylabel("Number of targets")
        ax2.set_title("Per-target coverage distribution", fontsize=10)
        cv = norm.std() / norm.mean() if norm.mean() else np.nan
        pct_low = (norm < 0.2).mean()
        pct_zero = (r["mean_depth"] == 0).mean()
        cv_text = (f"CV = {cv:.2f}\n"
                   f"{pct_low:.1%} of targets < 0.2\u00d7 mean depth\n"
                   f"{pct_zero:.2%} of targets at zero coverage\n"
                   f"n targets = {len(r):,}")
        ax2.text(0.98, 0.95, cv_text, transform=ax2.transAxes, fontsize=8.5,
                  ha="right", va="top",
                  bbox=dict(boxstyle="round", fc="white", ec="#cccccc", alpha=0.9))
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No mosdepth regions file (needs --by target.bed)",
                  ha="center", color=C_GREY)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)

    if inp.mosdepth_summary is not None:
        fig2 = plt.figure(figsize=A4_LANDSCAPE)
        ax = fig2.add_axes([0.05, 0.05, 0.9, 0.85])
        ax.axis("off")
        fig2.text(0.5, 0.95, "mosdepth Summary Table", ha="center", fontsize=13, fontweight="bold")
        s = inp.mosdepth_summary
        show_rows = s[s["chrom"].isin(["total", "total_region"])] if "chrom" in s.columns else s
        if len(show_rows) == 0:
            show_rows = s.tail(3)
        table = ax.table(cellText=show_rows.round(3).astype(str).values,
                          colLabels=show_rows.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1, 1.5)
        pdf.savefig(fig2)
        plt.close(fig2)


# ---- Page: GC bias ----------------------------------------------------------

def page_gc_bias(pdf, inp):
    if inp.mosdepth_regions is None or inp.fai is None:
        return

    r = inp.mosdepth_regions.copy()
    if len(r) > 20000:
        r = r.sample(20000, random_state=0).reset_index(drop=True)

    gcs = []
    for _, row in r.iterrows():
        seq = fetch_seq(inp.reference, inp.fai, row["chrom"], int(row["start"]), int(row["end"]))
        gcs.append(gc_fraction(seq) * 100 if seq else np.nan)
    r["gc"] = gcs
    r = r.dropna(subset=["gc"])
    if len(r) == 0:
        return

    mean_all = r["mean_depth"].mean()
    r["norm_depth"] = r["mean_depth"] / mean_all if mean_all else r["mean_depth"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=A4_LANDSCAPE)
    fig.suptitle("GC Bias (computed from reference + mosdepth per-target depth)",
                 fontsize=14, fontweight="bold")

    bins = np.arange(0, 105, 5)
    r["gc_bin"] = pd.cut(r["gc"], bins=bins, include_lowest=True)
    grouped = r.groupby("gc_bin", observed=True)["norm_depth"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 5]
    centers = [iv.mid for iv in grouped.index]
    ax1.plot(centers, grouped["mean"], marker="o", markersize=3, color=C_MAIN)
    ax1.axhline(1.0, color="#999999", linestyle="--", linewidth=1)
    ax1.set_xlabel("Target GC content (%)")
    ax1.set_ylabel("Mean normalized coverage")
    ax1.set_title("Coverage vs. GC content", fontsize=10)

    ax2.scatter(r["gc"], r["norm_depth"], s=3, alpha=0.25, color=C_MAIN)
    ax2.axhline(1.0, color="#999999", linestyle="--", linewidth=1)
    ax2.set_xlabel("Target GC content (%)")
    ax2.set_ylabel("Per-target normalized coverage")
    ax2.set_ylim(0, min(4, r["norm_depth"].quantile(0.995) + 0.5))
    ax2.set_title(f"Per-target scatter (n={len(r):,})", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)
    return r  # returned for reuse in the flags page


# ---- Page: variant overview --------------------------------------------------

def page_variant_overview(pdf, inp):
    if inp.vcf_df is None or len(inp.vcf_df) == 0:
        return
    v = inp.vcf_df

    fig, axes = plt.subplots(2, 2, figsize=A4_LANDSCAPE)
    fig.suptitle("Variant Overview (intrinsic, no truth set)", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    type_counts = v["type"].value_counts()
    bars = ax.bar(type_counts.index, type_counts.values, color=[C_MAIN, C_ACCENT, C_OK])
    for b, val in zip(bars, type_counts.values):
        ax.text(b.get_x() + b.get_width()/2, val, f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Variant counts by type", fontsize=10)

    snv = v[v["type"] == "SNV"]
    ti_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
    if len(snv):
        is_ti = snv.apply(lambda row: (row["ref"], row["alt"]) in ti_pairs, axis=1)
        titv = is_ti.sum() / max((~is_ti).sum(), 1)
    else:
        titv = np.nan
    ax = axes[0, 1]
    if v["qual"].notna().any():
        ax.hist(v["qual"].dropna(), bins=60, color=C_MAIN, alpha=0.85)
        ax.set_xlabel("QUAL")
        ax.set_ylabel("Count")
    ax.set_title(f"QUAL distribution  (Ti/Tv = {titv:.2f})" if pd.notna(titv) else "QUAL distribution",
                 fontsize=10)

    ax = axes[1, 0]
    filt_counts = v["filter"].value_counts().head(8)
    ax.bar(range(len(filt_counts)), filt_counts.values, color=C_ACCENT)
    ax.set_xticks(range(len(filt_counts)))
    ax.set_xticklabels(filt_counts.index.astype(str), rotation=30, ha="right", fontsize=7.5)
    ax.set_title("FILTER breakdown", fontsize=10)

    ax = axes[1, 1]
    indels = v[v["type"] == "INDEL"]
    if len(indels):
        lens = indels["indel_len"].clip(-20, 20)
        bins = np.arange(-20.5, 21.5, 1)
        ax.hist(lens, bins=bins, color=C_MAIN, alpha=0.85)
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_xlabel("Indel length (negative = deletion)")
        ax.set_ylabel("Count")
        frac_pm1 = ((indels["indel_len"].abs() == 1)).mean()
        ax.set_title(f"Indel size spectrum  ({frac_pm1:.0%} are \u00b11 bp)", fontsize=10)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No indels in VCF", ha="center", color=C_GREY)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ---- Page: VAF distribution + genome-wide variant density -------------------

def page_vaf_and_density(pdf, inp):
    if inp.vcf_df is None or len(inp.vcf_df) == 0:
        return
    v = inp.vcf_df

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=A4_LANDSCAPE)
    fig.suptitle("Allele Fraction & Genome-Wide Variant Density", fontsize=14, fontweight="bold")

    if v["vaf"].notna().any():
        ax1.hist(v["vaf"].dropna(), bins=60, range=(0, 1), color=C_MAIN, alpha=0.85)
        ax1.set_xlabel("VAF")
        ax1.set_ylabel("Variant count")
        ax1.set_title("VAF distribution (shape only -- compare between runs, "
                       "not to a truth set)", fontsize=10)
    else:
        ax1.axis("off")
        ax1.text(0.5, 0.5, "No AF/AD field found in VCF FORMAT/INFO", ha="center", color=C_GREY)

    if inp.fai is not None:
        chroms, offsets, total_len = build_genome_axis(inp.fai)
        v = v.copy()
        v["gx"] = v.apply(lambda row: genome_x(offsets, row["chrom"], row["pos"]), axis=1)
        v = v.dropna(subset=["gx"])
        if len(v):
            n_bins = 400
            bin_width = total_len / n_bins
            counts, edges = np.histogram(v["gx"], bins=n_bins, range=(0, total_len))
            centers = (edges[:-1] + edges[1:]) / 2
            ax2.bar(centers, counts, width=bin_width, color=C_MAIN)
            for name, length in chroms:
                ax2.axvline(offsets[name], color="#dddddd", linewidth=0.6)
            tick_pos = [offsets[n] + l / 2 for n, l in chroms]
            tick_lab = [n.replace("chr", "") for n, _ in chroms]
            ax2.set_xticks(tick_pos)
            ax2.set_xticklabels(tick_lab, fontsize=7)
            ax2.set_xlim(0, total_len)
            ax2.set_ylabel("Variants per bin")
            ax2.set_title("Variant density along the genome "
                           "(spikes can indicate mismapping / probe artifacts)", fontsize=10)
        else:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "Could not map variant chroms to reference contigs",
                      ha="center", color=C_GREY)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "No --reference given: cannot build genome-wide axis",
                  ha="center", color=C_GREY)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ---- Page: mapping/strand-bias annotation distributions ---------------------

def page_info_annotations(pdf, inp):
    if inp.vcf_df is None or len(inp.vcf_df) == 0:
        return
    v = inp.vcf_df
    present = [k for k in INFO_ANNOTATIONS_OF_INTEREST if k in v.columns and v[k].notna().any()]
    if not present:
        return

    n = len(present)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=A4_LANDSCAPE)
    fig.suptitle("Variant-Level Annotation Distributions (mapping/strand-bias QC)",
                 fontsize=14, fontweight="bold")
    axes = np.atleast_1d(axes).flatten()

    for i, key in enumerate(present):
        ax = axes[i]
        vals = v[key].dropna()
        ax.hist(vals, bins=50, color=C_MAIN, alpha=0.85)
        ax.set_title(key, fontsize=10)
        ax.set_xlabel(key)
    for i in range(len(present), len(axes)):
        axes[i].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


# ---- Page: FACETS CN profile -------------------------------------------------

def page_facets(pdf, inp):
    if inp.facets_cncf is None:
        return
    seg = inp.facets_cncf.copy()

    fig, axes = plt.subplots(2, 1, figsize=A4_LANDSCAPE, height_ratios=[2, 1])
    fig.suptitle("FACETS Copy-Number Segmentation", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    if inp.fai is not None:
        chroms, offsets, total_len = build_genome_axis(inp.fai)
        for _, row in seg.iterrows():
            gx1 = genome_x(offsets, str(row["chrom"]), row["start"])
            gx2 = genome_x(offsets, str(row["chrom"]), row["end"])
            if pd.isna(gx1) or pd.isna(gx2):
                continue
            ax1.plot([gx1, gx2], [row["cnlr"], row["cnlr"]], color=C_MAIN, linewidth=2.5)
        for name, length in chroms:
            ax1.axvline(offsets[name], color="#dddddd", linewidth=0.6)
        tick_pos = [offsets[n] + l / 2 for n, l in chroms]
        tick_lab = [n.replace("chr", "") for n, _ in chroms]
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_lab, fontsize=7)
        ax1.set_xlim(0, total_len)
    else:
        ax1.plot(range(len(seg)), seg["cnlr"], marker="o", markersize=2,
                  linestyle="none", color=C_MAIN)
        ax1.set_xlabel("Segment index (no reference given for genomic x-axis)")
    ax1.axhline(0, color="#999999", linewidth=1, linestyle="--")
    ax1.set_ylabel("log2 ratio (cnlr)")
    ax1.set_title(f"Genome-wide log-ratio profile ({len(seg)} segments)", fontsize=10)

    ax2 = axes[1]
    seg_len = (seg["end"] - seg["start"]).clip(lower=1)
    if "mafr" in seg.columns and seg["mafr"].notna().any():
        ax2.hist(seg["mafr"].dropna(), bins=40, color=C_ACCENT, alpha=0.85)
        ax2.set_xlabel("mafR (BAF variance per segment)")
        ax2.set_ylabel("Segment count")
        ax2.set_title("Segment BAF-variance distribution "
                       "(noisier = more allelic-imbalance noise from the assay)", fontsize=10)
    else:
        ax2.hist(np.log10(seg_len), bins=40, color=C_ACCENT, alpha=0.85)
        ax2.set_xlabel("log10(segment length, bp)")
        ax2.set_title("Segment length distribution", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)

    if inp.facets_summary:
        fig2 = plt.figure(figsize=(6, 3))
        ax = fig2.add_axes([0, 0, 1, 1])
        ax.axis("off")
        lines = [f"{k}: {v}" for k, v in inp.facets_summary.items()]
        ax.text(0.5, 0.5, "FACETS fit summary\n\n" + "\n".join(lines),
                 ha="center", va="center", fontsize=11)
        pdf.savefig(fig2)
        plt.close(fig2)


# ---- Page: CNAqc (generic passthrough) ---------------------------------------

def page_cnaqc(pdf, inp):
    if inp.cnaqc_table is None:
        return
    df = inp.cnaqc_table

    fig = plt.figure(figsize=A4_LANDSCAPE)
    fig.suptitle("CNAqc Table", fontsize=14, fontweight="bold")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        ax = fig.add_axes([0.08, 0.55, 0.84, 0.35])
        col = numeric_cols[0]
        ax.hist(df[col].dropna(), bins=40, color=C_MAIN, alpha=0.85)
        ax.set_title(f"Distribution of '{col}' (first numeric column found)", fontsize=10)
        ax.set_xlabel(col)

    ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.42])
    ax2.axis("off")
    preview = df.head(15)
    table = ax2.table(cellText=preview.astype(str).values, colLabels=preview.columns,
                       cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1, 1.2)
    if len(df) > 15:
        ax2.text(0.5, -0.05, f"... {len(df) - 15} more rows not shown",
                  transform=ax2.transAxes, ha="center", fontsize=7.5, color=C_GREY)

    fig.text(0.5, 0.965,
              "Rendered generically -- adapt parse_generic_table()/page_cnaqc() "
              "to your exact CNAqc export for tailored plots.",
              ha="center", fontsize=7.5, color=C_GREY)
    pdf.savefig(fig)
    plt.close(fig)


# ---- Page: automated heuristic flags -----------------------------------------

def page_flags(pdf, inp, gc_df):
    flags = []  # (severity, message)  severity in {"ok","warn","bad"}

    if inp.mosdepth_regions is not None:
        r = inp.mosdepth_regions
        mean_all = r["mean_depth"].mean()
        norm = r["mean_depth"] / mean_all if mean_all else r["mean_depth"]
        cv = norm.std() / norm.mean() if norm.mean() else np.nan
        pct_zero = (r["mean_depth"] == 0).mean()
        if pd.notna(cv):
            if cv > 0.6:
                flags.append(("bad", f"Per-target coverage CV is high ({cv:.2f}): "
                                       "possible capture uniformity problem."))
            elif cv > 0.4:
                flags.append(("warn", f"Per-target coverage CV is moderately elevated ({cv:.2f})."))
            else:
                flags.append(("ok", f"Per-target coverage CV looks reasonable ({cv:.2f})."))
        if pct_zero > 0.03:
            flags.append(("bad", f"{pct_zero:.1%} of targets have zero coverage: "
                                   "check for missing/failed baits."))
        elif pct_zero > 0.01:
            flags.append(("warn", f"{pct_zero:.2%} of targets have zero coverage."))

    if gc_df is not None and len(gc_df):
        mid = gc_df[(gc_df["gc"] >= 40) & (gc_df["gc"] <= 60)]["norm_depth"].mean()
        low = gc_df[gc_df["gc"] < 30]["norm_depth"].mean()
        high = gc_df[gc_df["gc"] > 70]["norm_depth"].mean()
        if pd.notna(mid) and mid > 0:
            for label, val in (("AT-rich (<30% GC)", low), ("GC-rich (>70% GC)", high)):
                if pd.notna(val):
                    ratio = val / mid
                    if ratio < 0.6:
                        flags.append(("bad", f"Strong GC bias: {label} targets average "
                                               f"only {ratio:.0%} of mid-GC coverage."))
                    elif ratio < 0.8:
                        flags.append(("warn", f"Mild GC bias: {label} targets average "
                                               f"{ratio:.0%} of mid-GC coverage."))

    if inp.vcf_df is not None and len(inp.vcf_df):
        v = inp.vcf_df
        snv = v[v["type"] == "SNV"]
        ti_pairs = {("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")}
        if len(snv):
            is_ti = snv.apply(lambda row: (row["ref"], row["alt"]) in ti_pairs, axis=1)
            titv = is_ti.sum() / max((~is_ti).sum(), 1)
            if titv < 2.0:
                flags.append(("warn", f"Ti/Tv ratio is low ({titv:.2f}) for exome data: "
                                        "check for FFPE-type C>T artifacts or low-confidence calls."))
            else:
                flags.append(("ok", f"Ti/Tv ratio looks typical ({titv:.2f})."))

        indels = v[v["type"] == "INDEL"]
        if len(indels) > 20:
            frac_pm1 = (indels["indel_len"].abs() == 1).mean()
            if frac_pm1 > 0.65:
                flags.append(("warn", f"{frac_pm1:.0%} of indels are \u00b11 bp: possible "
                                        "homopolymer/systematic indel bias -- check indel-prone "
                                        "capture/chemistry effects."))

    if inp.facets_cncf is not None:
        seg = inp.facets_cncf
        if "mafr" in seg.columns and seg["mafr"].notna().any():
            mean_mafr = seg["mafr"].mean()
            if mean_mafr > 0.15:
                flags.append(("warn", f"Mean FACETS mafR is elevated ({mean_mafr:.3f}): "
                                        "noisy allelic-imbalance signal, possibly assay-driven."))
        if len(seg) > 300:
            flags.append(("warn", f"FACETS produced an unusually high segment count ({len(seg)}): "
                                    "possible over-segmentation from noisy coverage."))

    fig = plt.figure(figsize=A4_LANDSCAPE)
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.85])
    ax.axis("off")
    fig.text(0.5, 0.95, "Automated Flags (heuristic -- investigate, don't over-trust)",
              ha="center", fontsize=14, fontweight="bold")

    color_map = {"ok": C_OK, "warn": C_WARN, "bad": C_BAD}
    icon_map = {"ok": "\u2713", "warn": "\u26a0", "bad": "\u2717"}
    y = 0.88
    if not flags:
        ax.text(0.05, y, "No inputs available for automated flagging "
                          "(need at least mosdepth regions or a VCF).",
                 fontsize=10, color=C_GREY)
    for sev, msg in flags:
        ax.text(0.03, y, icon_map[sev], fontsize=12, color=color_map[sev], fontweight="bold")
        ax.text(0.07, y, msg, fontsize=9.5, va="center")
        y -= 0.06

    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_report(args):
    inp = Inputs(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        page_title(pdf, inp, args)
        page_alignment_qc(pdf, inp)
        page_coverage(pdf, inp)
        gc_df = page_gc_bias(pdf, inp)
        page_variant_overview(pdf, inp)
        page_vaf_and_density(pdf, inp)
        page_info_annotations(pdf, inp)
        page_facets(pdf, inp)
        page_cnaqc(pdf, inp)
        page_flags(pdf, inp, gc_df)

    print(f"Wrote {out_path}")
    for w in inp.warnings:
        print(f"  warning: {w}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Single-sample capture QC report from Snakemake pipeline outputs "
                    "(mosdepth, samtools stats, VCF, FACETS) -- no truth set required.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--sample-name", required=True)
    ap.add_argument("--reference", help="Reference FASTA (indexed .fai used/created alongside)")
    ap.add_argument("--target-bed", help="Capture target BED (informational)")
    ap.add_argument("--mosdepth-prefix", help="Prefix passed to mosdepth (without file suffix)")
    ap.add_argument("--samtools-stats", help="Output of `samtools stats sample.bam`")
    ap.add_argument("--vcf", help="VCF or VCF.gz of variant calls")
    ap.add_argument("--facets-cncf", help="FACETS per-segment cncf output table")
    ap.add_argument("--facets-summary", help="FACETS purity/ploidy/dipLogR summary file")
    ap.add_argument("--cnaqc-table", help="Generic CNAqc-exported CSV/TSV table")
    ap.add_argument("--out", required=True, help="Output PDF path")
    args = ap.parse_args()
    build_report(args)


if __name__ == "__main__":
    main()