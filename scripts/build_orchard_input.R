#!/usr/bin/env Rscript --vanilla
# scripts/build_orchard_input.R
#
# Builds Orchard/Pairtree .ssm and .params.json from:
#   - DeepSomatic PASS VCF  (ref/alt counts)
#   - FACETS segments       (local CN for var_read_prob)
#   - PyClone6 results TSV  (cluster assignments)
#
# var_read_prob = 1 / total_cn (expected VAF of 1 mutant copy given local CN)
# Clipped to [0.01, 0.99] to avoid degenerate likelihoods.

suppressPackageStartupMessages({
  library(data.table)
  library(VariantAnnotation)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: build_orchard_input.R <vcf> <seg_tsv> <pyclone6_results> <out_ssm> <out_params> <sample_name>")
}

vcf_file     <- args[1]
seg_file     <- args[2]
pyclone_file <- args[3]
out_ssm      <- args[4]
out_params   <- args[5]
sname        <- args[6]

dir.create(dirname(out_ssm), showWarnings = FALSE, recursive = TRUE)

# ── PyClone6 cluster assignments ──────────────────────────────────────────────
pc     <- fread(pyclone_file)
pc_mut <- unique(pc[, .(mutation_id, cluster_id)])
message("PyClone6: ", pc_mut[, uniqueN(cluster_id)], " clusters, ",
        nrow(pc_mut), " mutations")

# ── FACETS segments for CN-corrected var_read_prob ────────────────────────────
seg <- fread(seg_file)
seg[, chrom_key  := paste0("chr", sub("^chr", "", as.character(chrom)))]
seg[, mcn_clean  := as.integer(ifelse(is.na(mcn), 1L, mcn))]
seg[, lcn_clean  := as.integer(ifelse(is.na(lcn), 1L, lcn))]

get_var_read_prob <- function(chrom, pos) {
  m <- seg[chrom_key == chrom & start <= pos & end >= pos]
  if (nrow(m) == 0L) return(0.5)
  total_cn <- m$mcn_clean[1L] + m$lcn_clean[1L]
  if (total_cn == 0L) return(0.5)
  max(0.01, min(0.99, 1.0 / total_cn))
}

# ── Parse VCF ─────────────────────────────────────────────────────────────────
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
chrom_v <- chrom_v[keep]; pos_v   <- pos_v[keep]
ref_v   <- ref_v[keep];   alt_v   <- alt_v[keep]
ref_c_v <- ref_c_v[keep]; alt_c_v <- alt_c_v[keep]



mutation_ids <- paste(chrom_v, pos_v, paste0(ref_v, ">", alt_v), sep = ":")
# mutation_id must match PyClone6 format exactly
#mutation_ids <- paste(sub("^chr", "", chrom_v), pos_v, ref_v, alt_v, sep = "_")

vcf_dt <- data.table(
  mutation_id = mutation_ids,
  chrom       = chrom_v,
  pos         = pos_v,
  var_reads   = as.integer(alt_c_v),
  total_reads = as.integer(ref_c_v + alt_c_v)
)

# ── Join with PyClone6 ────────────────────────────────────────────────────────
merged <- merge(vcf_dt, pc_mut, by = "mutation_id", all = FALSE)
message("Mutations matched to PyClone6: ", nrow(merged))
if (nrow(merged) < 5) {
  stop("Too few mutations matched — check mutation_id format between VCF and PyClone6")
}

merged[, var_read_prob := mapply(get_var_read_prob, chrom, pos)]

# ── Write .ssm ────────────────────────────────────────────────────────────────
ssm <- data.table(
  id            = paste0("s", seq_len(nrow(merged)) - 1L),
  name          = merged$mutation_id,
  var_reads     = merged$var_reads,
  total_reads   = merged$total_reads,
  var_read_prob = round(merged$var_read_prob, 4)
)

fwrite(ssm, out_ssm, sep = "\t", quote = FALSE)
message("SSM written: ", nrow(ssm), " mutations → ", out_ssm)


# ── Write .params.json ────────────────────────────────────────────────────────
cluster_ids   <- sort(unique(merged$cluster_id))

# Use a simple character vector for each cluster, NOT as.list()
clusters_list <- lapply(cluster_ids, function(cid) {
  ssm$id[merged$cluster_id == cid]
})

params <- list(
  samples  = sname, # Remove list() wrapper here to avoid [["sg070"]]
  clusters = clusters_list,
  garbage  = list()
)

# Use auto_unbox = TRUE to ensure single values (like sample name) 
# aren't forced into arrays, but lists (like clusters) remain arrays.
write(toJSON(params, auto_unbox = TRUE, pretty = TRUE), out_params)
