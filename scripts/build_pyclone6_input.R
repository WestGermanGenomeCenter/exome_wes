#!/usr/bin/env Rscript --vanilla
# scripts/build_pyclone6_input.R
# Usage: Rscript --vanilla scripts/build_pyclone6_input.R \
#            <vcf> <seg_tsv> <facets_rds> <out_tsv> <sample_name>
suppressPackageStartupMessages({
  library(data.table)
  library(VariantAnnotation)
})

args        <- commandArgs(trailingOnly = TRUE)
vcf_file    <- args[1]; seg_file <- args[2]; rds_file <- args[3]
out_tsv     <- args[4]; sname    <- args[5]

dir.create(dirname(out_tsv), showWarnings = FALSE, recursive = TRUE)

# ── Purity ────────────────────────────────────────────────────────────────────
obj    <- readRDS(rds_file)
purity <- round(obj$fit$purity, 4)
if (is.null(purity) || is.na(purity)) {
  warning("Purity NA — defaulting to 0.5"); purity <- 0.5
}
message("Purity: ", purity)

# ── CN segments ───────────────────────────────────────────────────────────────
seg <- fread(seg_file)
seg[, chrom_key := paste0("chr", sub("^chr", "", as.character(chrom)))]

get_cn <- function(chrom, pos) {
  m <- seg[chrom_key == chrom & start <= pos & end >= pos]
  if (nrow(m) == 0L) return(c(2L, 1L))   # PyClone6 diploid default: major=2, minor=1
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

keep    <- !is.na(ref_c_v) & !is.na(alt_c_v) & dp_v >= 25
message(sprintf("Mutations after depth filter: %d / %d", sum(keep), length(keep)))
if (sum(keep) < 10) stop("Fewer than 10 mutations after filtering")

chrom_v <- chrom_v[keep]; pos_v   <- pos_v[keep]
ref_v   <- ref_v[keep];   alt_v   <- alt_v[keep]
ref_c_v <- ref_c_v[keep]; alt_c_v <- alt_c_v[keep]

cn_mat  <- mapply(get_cn, chrom_v, pos_v)

# ── Write PyClone6 TSV ────────────────────────────────────────────────────────
# Required cols: mutation_id, sample_id, ref_counts, alt_counts,
#                normal_cn, minor_cn, major_cn, tumour_content
#tsv <- data.table(
#  mutation_id    = paste0(chrom_v, ":", pos_v, ":", ref_v, ">", alt_v),
#  sample_id      = sname,
#  ref_counts     = as.integer(ref_c_v),
#  alt_counts     = as.integer(alt_c_v),
#  normal_cn      = 2L,
#  minor_cn       = cn_mat[2, ],
#  major_cn       = cn_mat[1, ],
#  tumour_content = purity
#)


# ── Write PyClone6 TSV ────────────────────────────────────────────────────────
# Determine normal copy number dynamically based on sex chromosomes (assumes male baseline)
chrom_clean <- sub("^chr", "", chrom_v)
normal_cn_v <- ifelse(chrom_clean %in% c("X", "Y"), 1L, 2L)

tsv <- data.table(
  mutation_id    = paste0(chrom_v, ":", pos_v, ":", ref_v, ">", alt_v),
  sample_id      = sname,
  ref_counts     = as.integer(ref_c_v),
  alt_counts     = as.integer(alt_c_v),
  normal_cn      = normal_cn_v, # Dynamic fix
  minor_cn       = cn_mat[2, ],
  major_cn       = cn_mat[1, ],
  tumour_content = purity
)



fwrite(tsv, out_tsv, sep = "\t")
message("PyClone6 input written: ", nrow(tsv), " mutations → ", out_tsv)
