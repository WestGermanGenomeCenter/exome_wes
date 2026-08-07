#!/usr/bin/env bash
#
# offtarget_qc.sh -- classify somatic variant calls as on-target / off-target
# against an exome capture panel, and report depth (via mosdepth) + variant
# type for each call in one output table.
#
# Deliberately lean: intersect variants with target regions, check type,
# check depth, write one file. No background coverage sampling, no
# read-level specificity pass over the whole BAM -- if you need those,
# they're in the previous (fuller) version of this script; this one is
# built around "I expect a low number of variants, this is somatic
# calling" and optimizes for that.
#
set -euo pipefail

# ============================================================================
# Usage / argument parsing
# ============================================================================

usage() {
    cat << 'EOF'
Usage: offtarget_qc.sh -t targets.bed -b sample.bam -v variants.vcf(.gz) -g genome.fa.fai [OPTIONS]

Required:
  -t, --targets            Target BED file (any chrom order; sanity-checked and
                            sorted internally -- does not need to be pre-sorted)
  -b, --bam                Input BAM (indexed, or will be indexed in place)
  -v, --vcf                VCF or VCF.gz of variant calls
  -g, --genome-fai          Reference .fai (samtools faidx genome.fa)

Optional:
  -p, --pad                 Padding around targets in bp (default: 0 -- no
                            padding; off-target is a direct negation of the
                            real target regions)
  -o, --output-prefix       Output prefix (default: ./off_target)
  -s, --sample               Sample name in the VCF to use for FORMAT/DP if
                            INFO/DP isn't present (default: first sample)
      --pass-only            Only classify FILTER==PASS variants (default:
                            off, all records included; raw/unfiltered somatic
                            VCFs can have orders of magnitude more candidate
                            records than final PASS calls -- if your counts
                            look too high, this is usually why; consider
                            pointing at a PASS-filtered VCF instead)
      --min-mapq             Min mapping quality for mosdepth (default: 20)
      --canonical-regex      Regex (POSIX ERE) defining "canonical" contigs to
                            keep; alts/decoys/unplaced scaffolds are excluded
                            from analysis, with counts reported.
                            (default: '^(chr)?([0-9]{1,2}|X|Y|M|MT)$')
      --include-alt-contigs  Disable canonical-contig filtering.
      --threads              Threads for mosdepth/bedtools (default: 4)
  -h, --help                 Show this help
EOF
    exit "${1:-1}"
}

TARGETS="" ; BAM="" ; VCF="" ; GENOME_FAI="" ; PAD=0
OUTPUT_PREFIX="./off_target" ; VCF_SAMPLE="" ; PASS_ONLY=0
MIN_MAPQ=20 ; THREADS=4
CANONICAL_REGEX='^(chr)?([0-9]{1,2}|X|Y|M|MT)$' ; RESTRICT_CANONICAL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--targets) TARGETS="$2"; shift 2 ;;
        -b|--bam) BAM="$2"; shift 2 ;;
        -v|--vcf) VCF="$2"; shift 2 ;;
        -g|--genome-fai) GENOME_FAI="$2"; shift 2 ;;
        -p|--pad) PAD="$2"; shift 2 ;;
        -o|--output-prefix) OUTPUT_PREFIX="$2"; shift 2 ;;
        -s|--sample) VCF_SAMPLE="$2"; shift 2 ;;
        --pass-only) PASS_ONLY=1; shift ;;
        --min-mapq) MIN_MAPQ="$2"; shift 2 ;;
        --canonical-regex) CANONICAL_REGEX="$2"; shift 2 ;;
        --include-alt-contigs) RESTRICT_CANONICAL=0; shift ;;
        --threads) THREADS="$2"; shift 2 ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
done

for name_val in "targets:$TARGETS" "bam:$BAM" "vcf:$VCF" "genome-fai:$GENOME_FAI"; do
    name="${name_val%%:*}"; val="${name_val#*:}"
    if [[ -z "$val" ]]; then echo "Error: --$name is required" >&2; usage 1; fi
done

for tool in bedtools samtools bcftools tabix bgzip mosdepth awk sort; do
    command -v "$tool" >/dev/null 2>&1 || { echo "Error: required tool '$tool' not found in PATH" >&2; exit 1; }
done

for f in "$TARGETS" "$BAM" "$VCF" "$GENOME_FAI"; do
    [[ -s "$f" ]] || { echo "Error: input file missing or empty: $f" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUTPUT_PREFIX")" 2>/dev/null || true
WORKDIR="${OUTPUT_PREFIX}_work"
mkdir -p "$WORKDIR"
SUMMARY_FILE="${OUTPUT_PREFIX}_summary.txt"
: > "$SUMMARY_FILE"

log() { echo "[$(date +'%H:%M:%S')] $*" | tee -a "$SUMMARY_FILE.log" >&2; }

# every use of `head` truncating a pipeline is a SIGPIPE risk under
# `pipefail` (the upstream command gets killed, pipeline exit status is
# non-zero, `set -e` aborts the whole script -- silently, since neither
# side prints anything on SIGPIPE). This wraps any such pipeline safely.
safe_head() {
    local n="$1"
    ( set +o pipefail; head -n "$n" )
}

# ============================================================================
# Step 0: index the BAM / VCF if needed
# ============================================================================

if [[ ! -f "${BAM}.bai" && ! -f "${BAM%.bam}.bai" ]]; then
    log "Indexing BAM (not found)..."
    samtools index -@ "$THREADS" "$BAM"
fi

VCF_INDEXED="$VCF"
if [[ "$VCF" != *.gz ]]; then
    log "VCF is not bgzipped -- compressing and indexing a copy in the workdir..."
    VCF_INDEXED="$WORKDIR/$(basename "$VCF").gz"
    bgzip -c "$VCF" > "$VCF_INDEXED"
    tabix -p vcf "$VCF_INDEXED"
elif [[ ! -f "${VCF}.tbi" && ! -f "${VCF}.csi" ]]; then
    log "VCF is bgzipped but not indexed -- indexing..."
    tabix -p vcf "$VCF"
fi
VCF="$VCF_INDEXED"

# ============================================================================
# Step 0.5: restrict to canonical contigs (chr1-22/X/Y/M by default).
#   Real exome kits routinely bait some alt-haplotype regions (HLA etc),
#   and reference .fai files carry hundreds of decoy/alt/unplaced contigs
#   -- bedtools' -g genome-dictionary operations require every file being
#   compared to agree on the exact same contig set, so this is filtered
#   once, up front, and used for every subsequent -g call.
# ============================================================================

CANONICAL_FAI="$WORKDIR/canonical.fai"
if [[ "$RESTRICT_CANONICAL" -eq 1 ]]; then
    awk -v re="$CANONICAL_REGEX" '$1 ~ re' "$GENOME_FAI" > "$CANONICAL_FAI"
    N_FAI_TOTAL=$(wc -l < "$GENOME_FAI")
    N_FAI_CANONICAL=$(wc -l < "$CANONICAL_FAI")
    if [[ "$N_FAI_CANONICAL" -eq 0 ]]; then
        echo "Error: --canonical-regex '$CANONICAL_REGEX' matched 0 of $N_FAI_TOTAL contigs in $GENOME_FAI." >&2
        echo "  Sample contig names in your .fai: $(cut -f1 "$GENOME_FAI" | safe_head 5 | tr '\n' ' ')" >&2
        exit 1
    fi
    log "Step 0.5: restricting to canonical contigs: $N_FAI_CANONICAL / $N_FAI_TOTAL kept" \
        "($((N_FAI_TOTAL - N_FAI_CANONICAL)) alt/decoy/unplaced contigs excluded)"
    {
        echo "=== CANONICAL CONTIG FILTERING ==="
        echo "Regex used:                    $CANONICAL_REGEX"
        echo "Reference .fai contigs (total): $N_FAI_TOTAL"
        echo "Canonical contigs kept:         $N_FAI_CANONICAL"
        echo "Non-canonical contigs excluded: $((N_FAI_TOTAL - N_FAI_CANONICAL))"
        echo "  Excluded contig names (sample, up to 10):"
        comm -23 <(cut -f1 "$GENOME_FAI" | sort) <(cut -f1 "$CANONICAL_FAI" | sort) | safe_head 10 | sed 's/^/    /'
        echo ""
    } >> "$SUMMARY_FILE"
else
    cp "$GENOME_FAI" "$CANONICAL_FAI"
    log "Step 0.5: --include-alt-contigs set, using all $(wc -l < "$GENOME_FAI") contigs as-is (not recommended)"
fi
CANONICAL_CHROMS_CSV=$(cut -f1 "$CANONICAL_FAI" | paste -sd, -)

# ============================================================================
# Step 0.6: resolve VCF sample for FORMAT/DP fallback
# ============================================================================

if [[ -n "$VCF_SAMPLE" ]]; then
    if ! bcftools query -l "$VCF" | grep -qxF "$VCF_SAMPLE"; then
        echo "Error: sample '$VCF_SAMPLE' not found in $VCF. Samples present:" >&2
        bcftools query -l "$VCF" >&2
        exit 1
    fi
else
    VCF_SAMPLE=$(bcftools query -l "$VCF" | safe_head 1)
    if [[ -n "$VCF_SAMPLE" ]]; then
        log "No --sample given; defaulting to first VCF sample: '$VCF_SAMPLE'" \
            "(pass --sample explicitly if this is wrong, e.g. tumor vs. normal column order)"
    fi
fi

# ============================================================================
# Step 1: sanity-check and clean the target BED
# ============================================================================

log "Step 1: validating target BED ($TARGETS)..."

RAW_LINES=$(wc -l < "$TARGETS")

grep -vE '^(track|browser|#)' "$TARGETS" \
    | sed 's/\r$//' \
    | awk 'BEGIN{FS=OFS="\t"} NF>=3 {print $1,$2,$3}' \
    > "$WORKDIR/targets.stage1.bed"
N_AFTER_HEADER_STRIP=$(wc -l < "$WORKDIR/targets.stage1.bed")

awk 'BEGIN{FS=OFS="\t"} $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $2 < $3 && $2 >= 0' \
    "$WORKDIR/targets.stage1.bed" > "$WORKDIR/targets.stage2.bed"
N_MALFORMED=$((N_AFTER_HEADER_STRIP - $(wc -l < "$WORKDIR/targets.stage2.bed")))

cut -f1 "$WORKDIR/targets.stage2.bed" | sort -u > "$WORKDIR/bed_chroms.txt"
N_BED_CHROMS=$(wc -l < "$WORKDIR/bed_chroms.txt")
N_MATCHING_CHROMS=$(comm -12 "$WORKDIR/bed_chroms.txt" <(cut -f1 "$CANONICAL_FAI" | sort) | wc -l)

if [[ "$N_MATCHING_CHROMS" -eq 0 ]]; then
    echo "" >&2
    echo "Error: NONE of the target BED's chromosome names match the reference .fai." >&2
    echo "  BED chroms (sample):    $(safe_head 3 < "$WORKDIR/bed_chroms.txt" | tr '\n' ' ')" >&2
    echo "  This usually means a chr-prefix mismatch (chr1 vs 1) or the wrong genome build." >&2
    exit 1
elif [[ "$N_MATCHING_CHROMS" -lt "$N_BED_CHROMS" ]]; then
    log "  $((N_BED_CHROMS - N_MATCHING_CHROMS)) of $N_BED_CHROMS chrom names in the BED are non-canonical/unknown" \
        "and will be dropped."
fi

awk 'BEGIN{FS=OFS="\t"} NR==FNR{valid[$1]=1; next} ($1 in valid)' \
    "$CANONICAL_FAI" "$WORKDIR/targets.stage2.bed" > "$WORKDIR/targets.stage3.bed"

awk 'BEGIN{FS=OFS="\t"} NR==FNR{len[$1]=$2; next} ($1 in len) && $3<=len[$1]' \
    "$CANONICAL_FAI" "$WORKDIR/targets.stage3.bed" > "$WORKDIR/targets.stage4.bed"
N_OUT_OF_BOUNDS=$(( $(wc -l < "$WORKDIR/targets.stage3.bed") - $(wc -l < "$WORKDIR/targets.stage4.bed") ))

sort -u "$WORKDIR/targets.stage4.bed" > "$WORKDIR/targets.clean.bed"
N_DUP_REMOVED=$(( $(wc -l < "$WORKDIR/targets.stage4.bed") - $(wc -l < "$WORKDIR/targets.clean.bed") ))

N_CLEAN=$(wc -l < "$WORKDIR/targets.clean.bed")
if [[ "$N_CLEAN" -eq 0 ]]; then
    echo "Error: no valid target intervals remained after cleaning. Check $TARGETS manually." >&2
    exit 1
fi

{
    echo "=== TARGET BED SANITY CHECK ==="
    echo "Input rows:                       $RAW_LINES"
    echo "After header/comment strip:       $N_AFTER_HEADER_STRIP"
    echo "Dropped (malformed coords):       $N_MALFORMED"
    echo "Dropped (non-canonical/unknown):  $(( N_AFTER_HEADER_STRIP - N_MALFORMED - $(wc -l < "$WORKDIR/targets.stage3.bed") ))"
    echo "Dropped (out of bounds):          $N_OUT_OF_BOUNDS"
    echo "Dropped (exact duplicates):       $N_DUP_REMOVED"
    echo "Retained clean intervals:         $N_CLEAN"
    echo ""
} >> "$SUMMARY_FILE"
log "  retained $N_CLEAN / $RAW_LINES input rows after cleaning"

# ============================================================================
# Step 2: sort, optionally pad, merge -> the target region set everything
#   else is classified against.
# ============================================================================

if [[ "$PAD" -gt 0 ]]; then
    log "Step 2: sorting, padding (+/-${PAD}bp), and merging targets..."
    bedtools sort -g "$CANONICAL_FAI" -i "$WORKDIR/targets.clean.bed" > "$WORKDIR/targets.sorted.bed"
    bedtools slop -i "$WORKDIR/targets.sorted.bed" -g "$CANONICAL_FAI" -b "$PAD" > "$WORKDIR/targets.padded.bed"
else
    log "Step 2: sorting and merging targets (no padding)..."
    bedtools sort -g "$CANONICAL_FAI" -i "$WORKDIR/targets.clean.bed" > "$WORKDIR/targets.padded.bed"
fi
bedtools sort -g "$CANONICAL_FAI" -i "$WORKDIR/targets.padded.bed" \
    | bedtools merge -i - > "${OUTPUT_PREFIX}_target_regions.merged.bed"

N_PRE_MERGE=$(wc -l < "$WORKDIR/targets.padded.bed")
N_POST_MERGE=$(wc -l < "${OUTPUT_PREFIX}_target_regions.merged.bed")
if [[ "$N_POST_MERGE" -lt "$N_PRE_MERGE" ]]; then
    log "  merged $((N_PRE_MERGE - N_POST_MERGE)) overlapping intervals ($N_PRE_MERGE -> $N_POST_MERGE)"
fi

TARGET_BP=$(awk '{sum+=$3-$2} END{print sum+0}' "${OUTPUT_PREFIX}_target_regions.merged.bed")
{
    echo "=== TARGET REGIONS ==="
    echo "Padding applied (+/- bp): $PAD"
    echo "Merged target intervals:  $N_POST_MERGE"
    echo "Merged target bp:         $TARGET_BP"
    echo ""
} >> "$SUMMARY_FILE"

# ============================================================================
# Step 3: classify variants on-target / off-target (direct overlap /
#   no-overlap split against the target regions -- off-target is total
#   minus on-target by construction, not a separately-derived complement).
# ============================================================================

log "Step 3: classifying variants as on-target vs. off-target..."

FILTER_ARGS=()
if [[ "$PASS_ONLY" -eq 1 ]]; then
    FILTER_ARGS=(-f PASS,.)
    log "  --pass-only set: restricting to FILTER==PASS (or unset) records"
fi

N_VARS_RAW=$(bcftools view -H "$VCF" | wc -l)

if [[ "$RESTRICT_CANONICAL" -eq 1 ]]; then
    bcftools view "${FILTER_ARGS[@]}" -t "$CANONICAL_CHROMS_CSV" "$VCF" -Oz -o "$WORKDIR/variants.filtered.vcf.gz" 2>/dev/null \
        || bcftools view "${FILTER_ARGS[@]}" "$VCF" -Oz -o "$WORKDIR/variants.filtered.vcf.gz"
else
    bcftools view "${FILTER_ARGS[@]}" "$VCF" -Oz -o "$WORKDIR/variants.filtered.vcf.gz"
fi
tabix -f -p vcf "$WORKDIR/variants.filtered.vcf.gz"

bedtools intersect -a "$WORKDIR/variants.filtered.vcf.gz" \
    -b "${OUTPUT_PREFIX}_target_regions.merged.bed" -u -header \
    | bgzip -c > "${OUTPUT_PREFIX}_on_target_variants.vcf.gz"
tabix -f -p vcf "${OUTPUT_PREFIX}_on_target_variants.vcf.gz"

TOTAL_VARS=$(bcftools view -H "$WORKDIR/variants.filtered.vcf.gz" | wc -l)
ON_TARGET_VARS=$(bcftools view -H "${OUTPUT_PREFIX}_on_target_variants.vcf.gz" | wc -l)
OFF_TARGET_VARS=$((TOTAL_VARS - ON_TARGET_VARS))
N_VARS_NONCANONICAL=$((N_VARS_RAW - TOTAL_VARS))

{
    echo "=== VARIANT COUNTS ($( [[ $PASS_ONLY -eq 1 ]] && echo "PASS-only" || echo "all records" )) ==="
    echo "Total variants in VCF:        $N_VARS_RAW"
    echo "Excluded (non-canonical):     $N_VARS_NONCANONICAL"
    echo "Classified total:             $TOTAL_VARS"
    echo "  On-target:                  $ON_TARGET_VARS"
    echo "  Off-target:                 $OFF_TARGET_VARS"
    echo ""
} >> "$SUMMARY_FILE"
log "  on-target: $ON_TARGET_VARS  off-target: $OFF_TARGET_VARS  (total classified: $TOTAL_VARS," \
    "excluded non-canonical: $N_VARS_NONCANONICAL)"
if [[ "$TOTAL_VARS" -gt 50000 && "$PASS_ONLY" -eq 0 ]]; then
    log "  NOTE: $TOTAL_VARS is a lot of records for somatic calling -- if this looks too high," \
        "you may be pointing at a raw/unfiltered VCF; try --pass-only or a PASS-filtered VCF."
fi

# ============================================================================
# Step 4: build one combined table -- variant type, depth (mosdepth,
#   multithreaded), on-target/off-target bucket, distance-to-target.
# ============================================================================

log "Step 4: annotating variant type and depth (mosdepth, ${THREADS} threads)..."

bcftools query -f '%CHROM\t%POS0\t%POS\t%QUAL\t%FILTER\t%REF\t%ALT\n' "$WORKDIR/variants.filtered.vcf.gz" \
    | awk 'BEGIN{FS=OFS="\t"}
        {
            ref=$6; alt=$7;
            rl=length(ref); al=length(alt);
            if (rl==1 && al==1) { type="SNV" }
            else if (rl==al)    { type="MNV" }
            else if (al>rl)     { type="INS" }
            else                { type="DEL" }
            print $0, type
        }' \
    | bedtools sort -g "$CANONICAL_FAI" -i - > "$WORKDIR/variants.typed.tsv"

bedtools intersect -a "$WORKDIR/variants.typed.tsv" \
    -b "${OUTPUT_PREFIX}_target_regions.merged.bed" -c \
    > "$WORKDIR/variants.with_ontarget_flag.tsv"

bedtools closest -a "$WORKDIR/variants.with_ontarget_flag.tsv" \
    -b "${OUTPUT_PREFIX}_target_regions.merged.bed" -d \
    > "$WORKDIR/variants.with_distance.tsv"

cut -f1-3 "$WORKDIR/variants.with_distance.tsv" | sort -u \
    | bedtools sort -g "$CANONICAL_FAI" -i - > "$WORKDIR/variant_positions.bed"

N_POSITIONS=$(wc -l < "$WORKDIR/variant_positions.bed")
log "  running mosdepth on $N_POSITIONS distinct variant position(s)..."
mosdepth --no-per-base -x -Q "$MIN_MAPQ" -t "$THREADS" \
    --by "$WORKDIR/variant_positions.bed" \
    "$WORKDIR/mosdepth_variants" "$BAM"

zcat "$WORKDIR/mosdepth_variants.regions.bed.gz" > "$WORKDIR/variant_depths.tsv"

awk -F'\t' 'BEGIN{OFS="\t"}
    NR==FNR{depth[$1"\t"$3]=$4; next}
    {key=$1"\t"$3; print $0, (key in depth ? depth[key] : "NA")}' \
    "$WORKDIR/variant_depths.tsv" "$WORKDIR/variants.with_distance.tsv" \
    > "$WORKDIR/variants.final.tsv"

{
    echo -e "chrom\tstart\tend\tqual\tfilter\tref\talt\tvariant_type\ton_target_overlaps\ttarget_chrom\ttarget_start\ttarget_end\tdistance_to_target\tdepth"
    cat "$WORKDIR/variants.final.tsv"
} > "${OUTPUT_PREFIX}_variants_annotated.tsv"

N_OUT=$(( $(wc -l < "${OUTPUT_PREFIX}_variants_annotated.tsv") - 1 ))
log "Done. Wrote $N_OUT annotated variant(s) to ${OUTPUT_PREFIX}_variants_annotated.tsv"
log "Summary: $SUMMARY_FILE"
cat "$SUMMARY_FILE"