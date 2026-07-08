#!/usr/bin/env Rscript --vanilla
# scripts/build_phylogic_input.R
#
# Builds the PhylogicNDT input MAF only. No .sif file is written — the
# sample info string (id:maf:seg:purity) is built directly in the Snakemake
# shell command using the -s flag, since PhylogicNDT accepts that inline.
#
# Usage:
#   Rscript --vanilla scripts/build_phylogic_input.R \
#       <vcf> <seg_tsv> <facets_rds> <out_maf> <out_purity_txt> <sample_name>
#
# Can be run standalone for debugging:
#   Rscript --vanilla scripts/build_phylogic_input.R \
#       sample_somatic_pass.vcf.gz sample_cnv_segments.tsv sample_facets.rds \
#       sample.maf sample_purity.txt SAMPLE_001
suppressPackageStartupMessages({
  library(data.table)
  library(VariantAnnotation)
})

args        <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: build_phylogic_input.R <vcf> <seg> <rds> <out_maf> <out_purity_txt> <sample_name>")
}
vcf_file    <- args[1]; seg_file   <- args[2]; rds_file <- args[3]
out_maf     <- args[4]; out_purity <- args[5]; sname    <- args[6]

dir.create(dirname(out_maf), showWarnings = FALSE, recursive = TRUE)

# ── Purity from FACETS RDS ────────────────────────────────────────────────────
obj    <- readRDS(rds_file)
purity <- round(obj$fit$purity, 4)
if (is.null(purity) || is.na(purity)) {
  warning("Purity is NA — defaulting to 0.5"); purity <- 0.5
}
writeLines(as.character(purity), out_purity)
message("Purity: ", purity, " (written to ", out_purity, ")")

# ── CN segments ───────────────────────────────────────────────────────────────
seg <- fread(seg_file)
seg[, chrom_key := paste0("chr", sub("^chr", "", as.character(chrom)))]


# m <- seg[chrom_key == chrom & start <= pos & end >= pos]
get_cn <- function(chrom, pos) {
  m <- seg[chrom_key == chrom & start <= pos & end >= pos]
  if (nrow(m) == 0L) return(c(1L, 1L))   # diploid default
  major <- max(as.integer(m$mcn[1L]), 0L)
  minor <- max(as.integer(m$lcn[1L]), 0L)
  c(max(major, minor), min(major, minor))
}

# ── Parse VCF ────────────────────────────────────────────────────────────────
vcf     <- readVcf(vcf_file)
ad      <- geno(vcf)$AD
dp      <- geno(vcf)$DP
chrom_v <- as.character(seqnames(rowRanges(vcf)))
pos_v   <- start(rowRanges(vcf))
ref_v   <- as.character(rowRanges(vcf)$REF)
alt_v   <- sapply(rowRanges(vcf)$ALT, function(x) as.character(x[1]))
ref_c_v <- sapply(ad[, 1], `[`, 1)
alt_c_v <- sapply(ad[, 1], `[`, 2)
dp_v    <- dp[, 1]

keep    <- !is.na(ref_c_v) & !is.na(alt_c_v) & dp_v >= 10
message(sprintf("Mutations after depth filter: %d / %d", sum(keep), length(keep)))
if (sum(keep) < 10) stop("Fewer than 10 mutations after filtering — too few for PhylogicNDT")

chrom_v <- chrom_v[keep]; pos_v   <- pos_v[keep]
ref_v   <- ref_v[keep];   alt_v   <- alt_v[keep]
ref_c_v <- ref_c_v[keep]; alt_c_v <- alt_c_v[keep]

cn_mat  <- mapply(get_cn, chrom_v, pos_v)

# ── Write MAF ─────────────────────────────────────────────────────────────────
# PhylogicNDT required columns (--maf_input_type calc_ccf):
#   Hugo_Symbol, Chromosome, Start_position, Reference_Allele,
#   Tumor_Seq_Allele2, ref_count, alt_count, local_cn_a1, local_cn_a2
maf <- data.table(
  Hugo_Symbol       = "Unknown",
  Chromosome        = sub("^chr", "", chrom_v),
  Start_position    = pos_v,
  Reference_Allele  = ref_v,
  Tumor_Seq_Allele2 = alt_v,
  ref_count         = as.integer(ref_c_v),
  alt_count         = as.integer(alt_c_v),
  local_cn_a1       = cn_mat[1, ],
  local_cn_a2       = cn_mat[2, ]
)
fwrite(maf, out_maf, sep = "\t")
message("MAF written: ", nrow(maf), " mutations → ", out_maf)
