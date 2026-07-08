#!/usr/bin/env Rscript --vanilla
# scripts/run_facets.R
# Usage: Rscript --vanilla scripts/run_facets.R \
#            <pileup> <out_rds> <out_seg> <out_pdf> <out_qc> \
#            <sample_name> <cval> <min_nhet> <ndepth> <snp_nbhd> <genome>
suppressPackageStartupMessages({ library(facets); library(data.table) })

args        <- commandArgs(trailingOnly = TRUE)
pileup_file <- args[1];  out_rds  <- args[2];  out_seg <- args[3]
out_pdf     <- args[4];  out_qc   <- args[5];  sname   <- args[6]
cval        <- as.integer(args[7]); min_nhet <- as.integer(args[8])
ndepth      <- as.integer(args[9]); snp_nbhd <- as.integer(args[10])
genome      <- args[11]

dir.create(dirname(out_rds), showWarnings = FALSE, recursive = TRUE)

rcmat <- readSnpMatrix(pileup_file)
xx    <- preProcSample(rcmat, snp.nbhd = snp_nbhd, ndepth = ndepth, gbuild = genome)
oo    <- procSample(xx, cval = cval, min.nhet = min_nhet)
fit   <- emcncf(oo)

purity  <- round(fit$purity,  4)
ploidy  <- round(fit$ploidy,  4)
diplogr <- round(fit$dipLogR, 4)

saveRDS(list(sample = sname, rcmat = rcmat, xx = xx, oo = oo, fit = fit), out_rds)

seg <- as.data.table(fit$cncf)
seg[, sample := sname]
seg[, tcn    := fit$cncf$tcn.em]
seg[, lcn    := fit$cncf$lcn.em]
seg[, mcn    := tcn - lcn]
fwrite(seg, out_seg, sep = "\t")


qc <- c(
  paste0("sample\t",     sname),
  paste0("purity\t",     ifelse(is.na(purity), "NA", purity)),
  paste0("ploidy\t",     ifelse(is.na(ploidy), "NA", ploidy)),
  paste0("dipLogR\t",    ifelse(is.na(diplogr), "NA", diplogr)),
  paste0("n_segments\t", nrow(seg)),
  paste0("wgd\t",        ifelse(is.na(ploidy), "NA", ifelse(ploidy >= 3, "TRUE", "FALSE"))),
  if (is.na(purity))                "WARN\tPurity is NA — FACETS failed to converge on this sample" else NULL,
  if (!is.na(purity) && purity < 0.15)   "WARN\tPurity < 0.15 — CN fit unreliable" else NULL,
  if (nrow(seg) > 500)              "WARN\t>500 segments — consider raising cval" else NULL,
  if (!is.na(ploidy) && ploidy > 5)      "WARN\tPloidy > 5 — possible fit failure" else NULL
)








writeLines(qc, out_qc)

pdf(out_pdf, width = 14, height = 8)
plotSample(x = oo, emfit = fit, sname = sname, plot.type = "both")
dev.off()
