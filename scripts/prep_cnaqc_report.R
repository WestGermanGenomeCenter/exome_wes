#!/usr/bin/env Rscript
#
# extract_cnaqc_rds.R — Flatten a CNAqc RDS object into plain TSV/JSON files.
#
# CNAqc objects are R S3 lists containing nested tibbles, which cannot be
# reliably read from Python (pyreadr does not support this structure). This
# script does the one-time R-side extraction; cnaqc_report.py then builds
# the PDF entirely in Python from the flat files below.
#
# Usage:
#   Rscript extract_cnaqc_rds.R --rds sample_cnaqc.rds --outdir sample_cnaqc_export
#
# Output files (only written if the corresponding data exists in the RDS):
#   mutations.tsv          — per-mutation VAF/DP/NV/karyotype (+ CCF if computed)
#   cna_clonal.tsv          — clonal CNA segments
#   cna_subclonal.tsv       — subclonal CNA segments (if any)
#   karyotype_summary.tsv   — segment count / genome bp / mutation count per karyotype
#   peaks_matches.tsv       — CNAqc::analyze_peaks() results (if already run)
#   peaks_summary.tsv       — per-karyotype peak QC summary (if already run)
#   metadata.json           — sample, purity, ploidy, and other scalar fields
#
# Conda deps (conda-forge):
#   conda install -c conda-forge r-base r-optparse r-jsonlite

suppressWarnings(suppressPackageStartupMessages({
    library(optparse)
    library(jsonlite)
}))

option_list <- list(
    make_option("--rds",    type = "character", help = "CNAqc RDS file"),
    make_option("--outdir", type = "character", help = "Output directory")
)
opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$rds) || is.null(opt$outdir)) {
    stop("Usage: Rscript extract_cnaqc_rds.R --rds file.rds --outdir dir")
}
if (!file.exists(opt$rds)) {
    stop("RDS file not found: ", opt$rds)
}

dir.create(opt$outdir, recursive = TRUE, showWarnings = FALSE)

message("[extract_cnaqc] Loading RDS: ", opt$rds)
x <- readRDS(opt$rds)

if (!is.list(x)) {
    stop("Input RDS is not a list — does not look like a CNAqc object.")
}
if (!("cnaqc" %in% class(x))) {
    warning("[extract_cnaqc] Object class is not 'cnaqc' (found: ",
            paste(class(x), collapse = ", "), ") — attempting extraction anyway.")
}

write_tsv_safe <- function(df, path, label) {
    tryCatch({
        df <- as.data.frame(df)
        if (nrow(df) == 0) {
            message("[extract_cnaqc] ", label, " is empty — writing header-only file")
        }
        write.table(df, path, sep = "\t", quote = FALSE, row.names = FALSE)
        message("[extract_cnaqc] Wrote ", label, ": ", path, " (", nrow(df), " rows)")
    }, error = function(e) {
        warning("[extract_cnaqc] Could not write ", label, ": ", conditionMessage(e))
    })
}

# ── mutations ──────────────────────────────────────────────────────────────────
if (!is.null(x$mutations)) {
    write_tsv_safe(x$mutations, file.path(opt$outdir, "mutations.tsv"), "mutations")
} else {
    warning("[extract_cnaqc] No $mutations found in RDS — mutation-level plots will be unavailable")
}

# ── CNA segments (clonal) ─────────────────────────────────────────────────────
if (!is.null(x$cna)) {
    cna <- as.data.frame(x$cna)
    cna$subclonal <- FALSE
    write_tsv_safe(cna, file.path(opt$outdir, "cna_clonal.tsv"), "clonal CNA segments")
} else {
    warning("[extract_cnaqc] No $cna found in RDS — CNA segment plots will be unavailable")
}

# ── CNA segments (subclonal) ──────────────────────────────────────────────────
if (!is.null(x$cna_subclonal) && nrow(as.data.frame(x$cna_subclonal)) > 0) {
    cnas <- as.data.frame(x$cna_subclonal)
    cnas$subclonal <- TRUE
    write_tsv_safe(cnas, file.path(opt$outdir, "cna_subclonal.tsv"), "subclonal CNA segments")
} else {
    message("[extract_cnaqc] No subclonal CNA segments present (has_subclonal_CNA = ",
            isTRUE(x$has_subclonal_CNA), ")")
}

# ── karyotype summary ──────────────────────────────────────────────────────────
tryCatch({
    n_kar <- as.data.frame(x$n_karyotype)
    if (ncol(n_kar) == 2) colnames(n_kar) <- c("karyotype", "n_segments")

    bp_kar <- if (!is.null(x$basepairs_by_karyotype)) as.data.frame(x$basepairs_by_karyotype) else NULL

    mut_kar <- if (!is.null(x$mutations) && "karyotype" %in% colnames(x$mutations)) {
        as.data.frame(table(x$mutations$karyotype))
    } else NULL
    if (!is.null(mut_kar)) colnames(mut_kar) <- c("karyotype", "n_mutations")

    merged <- n_kar
    if (!is.null(bp_kar) && "karyotype" %in% colnames(bp_kar) && "n" %in% colnames(bp_kar)) {
        merged <- merge(merged, bp_kar[, c("karyotype", "n")], by = "karyotype", all = TRUE)
        colnames(merged)[colnames(merged) == "n"] <- "total_bp"
    }
    if (!is.null(mut_kar)) {
        merged <- merge(merged, mut_kar, by = "karyotype", all = TRUE)
    }
    write_tsv_safe(merged, file.path(opt$outdir, "karyotype_summary.tsv"), "karyotype summary")
}, error = function(e) {
    warning("[extract_cnaqc] Could not build karyotype summary: ", conditionMessage(e))
})

# ── peak analysis (optional — only present if CNAqc::analyze_peaks() was run) ──
if (!is.null(x$peaks_analysis)) {
    message("[extract_cnaqc] Peak analysis found in RDS — extracting")
    pa <- x$peaks_analysis
    if (!is.null(pa$matches)) {
        write_tsv_safe(pa$matches, file.path(opt$outdir, "peaks_matches.tsv"), "peak matches")
    } else {
        message("[extract_cnaqc] $peaks_analysis has no $matches element")
    }
    if (!is.null(pa$summary)) {
        write_tsv_safe(pa$summary, file.path(opt$outdir, "peaks_summary.tsv"), "peak summary")
    }
} else {
    message("[extract_cnaqc] No $peaks_analysis found in RDS. To enable official peak-QC plots, run:")
    message("    x <- CNAqc::analyze_peaks(x)")
    message("    saveRDS(x, 'sample_cnaqc_with_peaks.rds')")
    message("  then re-run this extractor on the updated RDS.")
}

# ── per-mutation CCF (optional — only present if CNAqc::compute_CCF() was run) ─
has_ccf <- !is.null(x$mutations) && "CCF" %in% colnames(x$mutations)
if (has_ccf) {
    message("[extract_cnaqc] Per-mutation CCF column found in $mutations")
} else {
    message("[extract_cnaqc] No per-mutation CCF column found. To compute it, run:")
    message("    x <- CNAqc::compute_CCF(x)")
    message("    saveRDS(x, 'sample_cnaqc_with_ccf.rds')")
    message("  then re-run this extractor on the updated RDS.")
}

# ── metadata ───────────────────────────────────────────────────────────────────
safe_get <- function(x, name) if (!is.null(x[[name]])) x[[name]] else NA

meta <- list(
    sample                    = safe_get(x, "sample"),
    reference_genome          = safe_get(x, "reference_genome"),
    purity                    = safe_get(x, "purity"),
    ploidy                    = safe_get(x, "ploidy"),
    n_mutations               = safe_get(x, "n_mutations"),
    n_cna                     = safe_get(x, "n_cna"),
    n_cna_clonal              = safe_get(x, "n_cna_clonal"),
    n_cna_subclonal           = safe_get(x, "n_cna_subclonal"),
    has_subclonal_CNA         = safe_get(x, "has_subclonal_CNA"),
    most_prevalent_karyotype  = safe_get(x, "most_prevalent_karyotype"),
    most_mutations_karyotype  = safe_get(x, "most_mutations_karyotype"),
    has_peaks_analysis        = !is.null(x$peaks_analysis),
    has_ccf                   = has_ccf
)
write(toJSON(meta, auto_unbox = TRUE, pretty = TRUE, null = "null"),
      file.path(opt$outdir, "metadata.json"))
message("[extract_cnaqc] Wrote metadata.json")

message("[extract_cnaqc] Done. Files written to: ", opt$outdir)
