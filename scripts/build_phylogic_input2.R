#!/usr/bin/env Rscript --vanilla
# scripts/build_phylogic_input.R
#
# Builds PhylogicNDT input MAF WITH pre-computed CCF histograms.
#
suppressPackageStartupMessages({
  library(data.table)
  library(VariantAnnotation)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 7) {
  stop("Usage: build_phylogic_input.R <vcf> <seg> <rds> <out_maf> <out_seg> <out_purity_txt> <sample_name>")
}

vcf_file   <- args[1]
seg_file   <- args[2]
rds_file   <- args[3]
out_maf    <- args[4]
out_seg    <- args[5]
out_purity <- args[6]
sname      <- args[7]

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

## PhylogicNDT timing-format requires exactly: ID Chromosome Start_position End_Position A1_CN A2_CN
#phylogic_seg <- data.table(
#  ID             = sname,
#  Chromosome     = sub("^chr", "", as.character(seg$chrom)),
#  Start_position = as.integer(seg$start),
#  End_Position   = as.integer(seg$end),
#  A1_CN          = pmax(as.integer(seg$mcn), as.integer(seg$lcn)),
#  A2_CN          = pmin(as.integer(seg$mcn), as.integer(seg$lcn))
#)
phylogic_seg <- data.table(
  Chromosome = sub("^chr", "", as.character(seg$chrom)),
  Start      = as.integer(seg$start),
  End        = as.integer(seg$end),
  A1.Seg.CN  = pmax(0L, as.integer(ifelse(is.na(seg$mcn), 0L, seg$mcn))),
  A2.Seg.CN  = pmax(0L, as.integer(ifelse(is.na(seg$lcn), 0L, seg$lcn)))
)


# timing_format parser has no 23/24 remapping unlike alleliccapseg
phylogic_seg[Chromosome == "23", Chromosome := "X"]
phylogic_seg[Chromosome == "24", Chromosome := "Y"]

fwrite(phylogic_seg, out_seg, sep = "\t", quote = FALSE)
message("Timing-format CN segments written to ", out_seg)

# ── Build chrom lookup for per-mutation CN annotation ─────────────────────────
seg[, chrom_key := paste0("chr", sub("^chr", "", as.character(chrom)))]

get_cn <- function(chrom, pos) {
  m <- seg[chrom_key == chrom & start <= pos & end >= pos]
  if (nrow(m) == 0L) return(c(1L, 1L))
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
if (sum(keep) < 10) stop("Fewer than 10 mutations after filtering — too few for PhylogicNDT")

chrom_v <- chrom_v[keep]; pos_v   <- pos_v[keep]
ref_v   <- ref_v[keep];   alt_v   <- alt_v[keep]
ref_c_v <- ref_c_v[keep]; alt_c_v <- alt_c_v[keep]

cn_mat <- mapply(get_cn, chrom_v, pos_v)

# ── Compute CCF histograms ────────────────────────────────────────────────────
compute_ccf_histogram <- function(alt_count, total_depth, purity, cn_a1, cn_a2, grid_size = 101) {
  ccf_grid <- seq(0, 1, length.out = grid_size)
  total_cn <- cn_a1 + cn_a2
  if (total_cn == 0) total_cn <- 2
  expected_vaf <- (purity * ccf_grid) / ((purity * total_cn) + ((1 - purity) * 2))
  expected_vaf <- pmin(pmax(expected_vaf, 0), 1)
  likelihood <- dbinom(alt_count, size = total_depth, prob = expected_vaf)
  if (sum(likelihood) == 0) {
    rep(1 / grid_size, grid_size)
  } else {
    likelihood / sum(likelihood)
  }
}

grid_size     <- 101
total_depth_v <- ref_c_v + alt_c_v

ccf_histograms <- mapply(
  compute_ccf_histogram,
  alt_c_v, total_depth_v, purity, cn_mat[1, ], cn_mat[2, ],
  MoreArgs = list(grid_size = grid_size),
  SIMPLIFY = FALSE
)

ccf_cols  <- as.data.table(do.call(rbind, ccf_histograms))
ccf_names <- c("ccf_raw_0", sprintf("ccf_raw_%.2f", seq(0.01, 0.99, by = 0.01)), "ccf_raw_1")
setnames(ccf_cols, ccf_names)

# ── Write MAF ─────────────────────────────────────────────────────────────────
maf <- data.table(
  Hugo_Symbol          = "Unknown",
  Chromosome           = sub("^chr", "", chrom_v),
  Start_position       = pos_v,
  Reference_Allele     = ref_v,
  Tumor_Seq_Allele2    = alt_v,
  Tumor_Sample_Barcode = sname,
  ref_count            = as.integer(ref_c_v),
  alt_count            = as.integer(alt_c_v),
  local_cn_a1          = cn_mat[1, ],
  local_cn_a2          = cn_mat[2, ]
)

maf <- cbind(maf, ccf_cols)
fwrite(maf, out_maf, sep = "\t")
message("MAF written: ", nrow(maf), " mutations with ", grid_size, " CCF bins → ", out_maf)