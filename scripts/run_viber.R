#!/usr/bin/env Rscript --vanilla

suppressPackageStartupMessages({
    library(VIBER)
    library(data.table)
    library(ggplot2)
})


args <- commandArgs(trailingOnly=TRUE)

if (length(args) < 3) {
    stop(
        paste(
            "Usage:",
            "run_viber.R <input.tsv> <output_prefix> <output_dir>"
        )
    )
}


input <- args[1]
prefix <- args[2]
outdir <- args[3]


dir.create(
    outdir,
    recursive=TRUE,
    showWarnings=FALSE
)


# ---------------------------------------------------------
# Load input
# ---------------------------------------------------------

#dat <- fread(input)
# ---------------------------------------------------------
# Load input
# ---------------------------------------------------------
dat <- fread(input)

# FIX: Filter for copy-number neutral diploid regions to avoid CNV-driven artifacts
# This ensures successes/trials represent true evolutionary CCF shifts.
if ("local_cn_a1" %in% names(dat) & "local_cn_a2" %in% names(dat)) {
    message("Filtering mutations to copy-number neutral regions (1+1 diploid)...")
    dat <- dat[local_cn_a1 == 1 & local_cn_a2 == 1]
    message("Mutations remaining for VIBER clustering: ", nrow(dat))
} else {
    warning("No copy-number columns found! VIBER will cluster on raw VAF, which may create false subclones.")
}

required <- c(
    "mutation_id",
    "successes",
    "trials"
)

missing <- setdiff(required, names(dat))

if (length(missing) > 0) {
    stop(
        "Missing required columns: ",
        paste(missing, collapse=", ")
    )
}


if (nrow(dat) < 10) {
    stop(
        "Too few mutations for VIBER: ",
        nrow(dat)
    )
}


if (any(is.na(dat$successes)) ||
    any(is.na(dat$trials))) {

    stop("NA values detected in counts")
}


if (any(dat$successes < 0)) {
    stop("Negative alternate counts detected")
}


if (any(dat$trials <= 0)) {
    stop("Zero/negative trials detected")
}


if (any(dat$successes > dat$trials)) {
    stop(
        "Found mutations where alt count > total depth"
    )
}


message(
    "Running VIBER on ",
    nrow(dat),
    " mutations"
)


# ---------------------------------------------------------
# Prepare matrices
# ---------------------------------------------------------

successes <- matrix(
    dat$successes,
    ncol=1
)

trials <- matrix(
    dat$trials,
    ncol=1
)


colnames(successes) <- "S1"
colnames(trials) <- "S1"


# ---------------------------------------------------------
# Fit
# ---------------------------------------------------------

fit <- variational_fit(
    successes,
    trials
)


fit <- choose_clusters(
    fit,
    binomial_cutoff = 0,
    dimensions_cutoff = 0,
    pi_cutoff = 0.02
)

saveRDS(fit, file = paste0(prefix, "_viber_fit.rds"))


# ---------------------------------------------------------
# Cluster assignments
# ---------------------------------------------------------

clusters <- data.table(
    mutation_id = dat$mutation_id,
    cluster = fit$clusters
)

fwrite(
    clusters,
    paste0(prefix, "_viber_clusters.tsv"),
    sep="\t"
)



# ---------------------------------------------------------
# Plots
# ---------------------------------------------------------

png(paste0(prefix, "_clusters.png"), width=1200, height=1000)
# ---------------------------------------------------------
# Custom VAF Distribution Plot (Better for single sample)
# ---------------------------------------------------------
library(ggplot2)

# Create a dataframe for plotting
plot_df <- data.table(
    mutation_id = dat$mutation_id,
    vaf = dat$successes / dat$trials,
    cluster = as.factor(fit$clusters)
)

p_vaf <- ggplot(plot_df, aes(x = vaf, color = cluster)) +
    geom_histogram(binwidth = 0.02, alpha = 0.6, position = "identity") +
    theme_minimal() +
    labs(title = paste("VAF Distribution by Cluster -", prefix),
         x = "Variant Allele Frequency (VAF)",
         y = "Count",
         color = "Cluster") +
    scale_color_discrete(name = "Cluster")

ggsave(
    filename = paste0(prefix, "_vaf_distribution.png"),
    plot = p_vaf,
    width = 8, height = 6, dpi = 300
)
# inserted after some feedback

#dev.off()



png(paste0(prefix, "_mixing_proportions.png"), width=1200, height=1000)

print(
    plot_mixing_proportions(fit)
)

dev.off()

png(paste0(prefix, "_latent_variables.png"), width=1200, height=1000)

print(
    plot_latent_variables(fit)
)

dev.off()


png(paste0(prefix, "_ELBO.png"), width=1200, height=1000)

print(
    plot_ELBO(fit)
)

dev.off()

png(paste0(prefix, "_peaks.png"), width=1200, height=1000)

print(
    plot_peaks(fit)
)

dev.off()


message("VIBER completed")