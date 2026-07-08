#!/usr/bin/env Rscript --vanilla
# scripts/run_cnaqc.R
# Usage:
#   Rscript --vanilla scripts/run_cnaqc.R \
#       <vcf> <seg_tsv> <facets_rds> <out_qc> <out_plot> <out_rds> \
#       <sample_name> <purity_tolerance>
suppressPackageStartupMessages({
  library(CNAqc)
  library(data.table)
  library(VariantAnnotation)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Usage: run_cnaqc.R <vcf> <seg> <rds> <out_qc> <out_plot> <out_rds> <sample_name> <purity_tol>")
}
vcf_file   <- args[1]; seg_file <- args[2]; rds_file  <- args[3]
out_qc     <- args[4]; out_plot <- args[5]; out_rds   <- args[6]
sample_name<- args[7]; purity_tol <- as.numeric(args[8])

dir.create(dirname(out_qc), showWarnings = FALSE, recursive = TRUE)

write_fail <- function(msg) {
  message("WARN: ", msg)
  writeLines(c(
    paste0("sample\t", sample_name),
    "cnaqc_pass\tNA",
    paste0("WARN\t", msg)
  ), out_qc)
  pdf(out_plot); plot.new(); text(0.5, 0.5, msg); dev.off()
  saveRDS(NULL, out_rds)
  quit(save = "no", status = 0)
}
# ── Purity from FACETS RDS ────────────────────────────────────────────────────
obj    <- readRDS(rds_file)
purity <- obj$fit$purity
if (is.null(purity) || is.na(purity)) {
  write_fail("FACETS purity is NA — cannot run CNAqc on this sample")
}
message("Purity: ", round(purity, 3))

# ── Load CN segments and validate allele-specific CN is usable ───────────────
seg <- fread(seg_file)
required_cols <- c("chrom", "start", "end", "mcn", "lcn")
#equired_cols <- c("chrom", "start", "end", "mcn", "lcn")
# required_cols <- c("chrom", "loc.start", "loc.end", "mcn", "lcn")
missing_cols  <- setdiff(required_cols, names(seg))
if (length(missing_cols) > 0) {
  write_fail(paste0("Segment file missing columns: ", paste(missing_cols, collapse = ", ")))
}

mcn_num <- suppressWarnings(as.integer(seg$mcn))
lcn_num <- suppressWarnings(as.integer(seg$lcn))

if (nrow(seg) == 0 || all(is.na(mcn_num)) || all(is.na(lcn_num))) {
  write_fail(paste0(
    "FACETS produced no usable allele-specific CN (mcn/lcn empty) — ",
    "likely too few heterozygous SNPs per segment relative to min_nhet. ",
    "Check pileup density and min_nhet in config."
  ))
}


# to this:
cna <- data.frame(
  chr   = paste0("chr", sub("^chr", "", as.character(seg$chrom))),
  from  = seg$start,
  to    = seg$end,
  Major = pmax(mcn_num, 0L),
  minor = pmax(lcn_num, 0L)
)
#cna <- data.frame(
#  chr   = paste0("chr", sub("^chr", "", as.character(seg$chrom))),
#  from  = seg$loc.start,
#  to    = seg$loc.end,
#  Major = pmax(mcn_num, 0L),
#  minor = pmax(lcn_num, 0L)
#)
cna <- cna[!is.na(cna$Major) & !is.na(cna$minor), ]

if (nrow(cna) == 0) {
  write_fail("All CN segments had NA major/minor CN after cleaning — nothing usable for CNAqc")
}

# ── Load somatic mutations ─────────────────────────────────────────────────────
vcf  <- readVcf(vcf_file)
ad   <- geno(vcf)$AD
dp   <- geno(vcf)$DP
snvs <- data.frame(
  chr  = as.character(seqnames(rowRanges(vcf))),
  from = start(rowRanges(vcf)),
  to   = start(rowRanges(vcf)),
  ref  = as.character(rowRanges(vcf)$REF),
  alt  = sapply(rowRanges(vcf)$ALT, function(x) as.character(x[1])),
  NV   = sapply(ad[, 1], `[`, 2),
  DP   = dp[, 1]
)
snvs <- snvs[!is.na(snvs$NV) & snvs$DP >= 10, ]
snvs$VAF <- snvs$NV / snvs$DP

if (nrow(snvs) < 20) {
  write_fail(paste0("Only ", nrow(snvs), " mutations with depth >= 10 — CNAqc unreliable (need >= 20)"))
}

# ── Build and run CNAqc ────────────────────────────────────────────────────────
x <- tryCatch(
  CNAqc::init(mutations = snvs, cna = cna, purity = purity, sample = sample_name),
  error = function(e) { message("CNAqc init error: ", e$message); NULL }
)
if (is.null(x)) write_fail("CNAqc::init() failed — check segment/mutation format")

x <- tryCatch(
  CNAqc::analyze_peaks(x, epsilon = purity_tol),
  error = function(e) { message("analyze_peaks error: ", e$message); x }
)

qc_pass  <- tryCatch(CNAqc::QC_passed(x), error = function(e) NA)
lambda   <- tryCatch(round(CNAqc::score(x), 4), error = function(e) NA)
n_mapped <- tryCatch(nrow(CNAqc::mutations(x)), error = function(e) NA)

message(sprintf("CNAqc: PASS=%s  lambda=%s  n_mapped=%s",
                qc_pass, ifelse(is.na(lambda), "NA", lambda), n_mapped))

# ── Write QC ──────────────────────────────────────────────────────────────────
warn <- character(0)
if (isFALSE(qc_pass))
  warn <- c(warn, paste0("WARN\tlambda=", lambda, " > epsilon=", purity_tol,
                         " — CN/purity solution inconsistent with VAF distribution"))
if (purity < 0.15)
  warn <- c(warn, "WARN\tPurity < 0.15 — CNAqc results unreliable")

writeLines(c(
  paste0("sample\t",       sample_name),
  paste0("purity_used\t",  round(purity, 4)),
  paste0("n_mutations\t",  nrow(snvs)),
  paste0("n_mapped\t",     n_mapped),
  paste0("lambda_score\t", lambda),
  paste0("epsilon\t",      purity_tol),
  paste0("cnaqc_pass\t",   ifelse(is.na(qc_pass), "NA", ifelse(qc_pass, "PASS", "FAIL"))),
  warn
), out_qc)

saveRDS(x, out_rds)

# ── Plot ───────────────────────────────────────────────────────────────────────
pdf(out_plot, width = 14, height = 8)
tryCatch({
  print(CNAqc::plot_peaks_analysis(x))
  print(CNAqc::plot_data_histogram(x))
}, error = function(e) { plot.new(); text(0.5, 0.5, paste("Plot error:", e$message)) })
dev.off()

message("CNAqc complete.")