#!/usr/bin/env bash
# scripts/somatic_qc.sh
# Usage: somatic_qc.sh <vcf> <out_qc> <out_stats> <sample_name>
#
# Computes SNV/indel counts, Ti/Tv ratio, VAF and depth percentiles
# from a (PASS-filtered) somatic VCF. No awk - uses bcftools, sort,
# and datamash for all numeric work.
set -euo pipefail

VCF="$1"
OUT_QC="$2"
OUT_STATS="$3"
SAMPLE="$4"

mkdir -p "$(dirname "$OUT_QC")"

# ── 1. Full bcftools stats (also used by MultiQC) ────────────────────────────
bcftools stats "$VCF" > "$OUT_STATS"

# ── 2. Pull SNV/indel/Ti-Tv lines straight out of bcftools stats ────────────
{
    echo "=== Somatic QC: ${SAMPLE} ==="
    grep "^SN"   "$OUT_STATS"
    grep "^TSTV" "$OUT_STATS"

    n_tot=$(grep "number of records:" "$OUT_STATS" | cut -f4)
    if [ -n "${n_tot:-}" ] && [ "$n_tot" -lt 50 ]; then
        printf "WARN\t<50 PASS variants - clustering unreliable\n"
    fi
    echo ""
} > "$OUT_QC"

# ── 3. VAF percentiles via datamash (no awk) ─────────────────────────────────
vaf_file=$(mktemp)
bcftools query -f '[%VAF]\n' "$VCF" > "$vaf_file"
n_vaf=$(wc -l < "$vaf_file")

{
    echo "--- VAF percentiles ---"
    if [ "$n_vaf" -eq 0 ]; then
        printf "WARN\tNo AF values found\n"
    else
        sort -n "$vaf_file" | datamash perc:10 1 perc:25 1 perc:50 1 perc:75 1 perc:90 1 \
            | tr '\t' '\n' \
            | paste -d'\t' <(printf "P10\nP25\nP50\nP75\nP90\n") -

        median_vaf=$(sort -n "$vaf_file" | datamash perc:50 1)
        if [ "$(echo "$median_vaf > 0.6" | bc -l)" = "1" ]; then
            printf "WARN\tMedian VAF>0.6 - germline contamination?\n"
        fi
    fi
    echo ""
} >> "$OUT_QC"
rm -f "$vaf_file"

# ── 4. Depth percentiles via datamash (no awk) ───────────────────────────────
dp_file=$(mktemp)
bcftools query -f '[%DP]\n' "$VCF" > "$dp_file"
n_dp=$(wc -l < "$dp_file")

{
    echo "--- Depth percentiles ---"
    if [ "$n_dp" -eq 0 ]; then
        printf "WARN\tNo DP values found\n"
    else
        sort -n "$dp_file" | datamash perc:10 1 perc:50 1 perc:90 1 \
            | tr '\t' '\n' \
            | paste -d'\t' <(printf "P10\nP50\nP90\n") -
    fi
} >> "$OUT_QC"
rm -f "$dp_file"

echo "Somatic QC written: $OUT_QC"