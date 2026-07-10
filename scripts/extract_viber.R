#!/usr/bin/env Rscript

# Extract useful tables from a VIBER vb_bmm RDS object
#
# Usage:
#
# Rscript extract_viber_rds.R \
#     --rds st015_viber_fit.rds \
#     --outdir st015_viber_export
#
suppressWarnings({
    library(optparse)
    library(data.table)
})


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

option_list <- list(

    make_option(
        c("--rds"),
        type="character",
        help="VIBER RDS file",
        metavar="FILE"
    ),

    make_option(
        c("--input"),
        type="character",
        help="Original VIBER input TSV containing mutation_id, successes, trials",
        metavar="FILE"
    ),

    make_option(
        c("--outdir"),
        type="character",
        help="Output directory",
        metavar="DIR"
    )
)


opt <- parse_args(
    OptionParser(
        option_list=option_list
    )
)


if (is.null(opt$rds) ||
    is.null(opt$input) ||
    is.null(opt$outdir)) {

    stop(
        paste(
            "Usage:",
            "Rscript extract_viber_rds.R",
            "--rds file.rds",
            "--input viber_input.tsv",
            "--outdir directory"
        )
    )

}


dir.create(
    opt$outdir,
    recursive=TRUE,
    showWarnings=FALSE
)



message("[extract_viber_rds] Loading VIBER object...")

fit <- readRDS(
    opt$rds
)


message("[extract_viber_rds] Loading input TSV...")

input <- fread(
    opt$input
)


if (!"mutation_id" %in% names(input)) {
    stop(
        "Input TSV does not contain mutation_id column"
    )
}

# ---------------------------------------------------------
# Mutations with cluster assignments
# ---------------------------------------------------------

message("[extract_viber_rds] Writing mutations.tsv")


clusters <- as.character(
    fit$labels$cluster.Binomial
)


if (length(clusters) != nrow(input)) {
    stop(
        "Number of cluster labels does not match number of input mutations"
    )
}

clusters <- as.character(
    fit$labels$cluster.Binomial
)

# safety check
if (any(is.na(clusters))) {

    warning(
        "Found ",
        sum(is.na(clusters)),
        " mutations without cluster assignment. Recovering from posterior."
    )

    rnk <- as.matrix(
        fit$r_nk
    )

    clusters[is.na(clusters)] <- colnames(rnk)[
        max.col(
            rnk[is.na(clusters), , drop=FALSE],
            ties.method="first"
        )
    ]
}


mutations <- data.frame(
    mutation_id = input$mutation_id,
    mutation_index = seq_len(nrow(input)),
    cluster = clusters,
    successes = input$successes,
    trials = input$trials
)

write.table(
    mutations,
    file.path(opt$outdir, "mutations.tsv"),
    sep="\t",
    quote=FALSE,
    row.names=FALSE
)


# ---------------------------------------------------------
# Cluster parameters
# ---------------------------------------------------------

message("[extract_viber_rds] Writing parameters.tsv")


parameters <- data.frame(

    cluster =
        names(fit$pi_k),

    pi =
        as.numeric(fit$pi_k),

    theta =
        as.numeric(fit$theta_k)

)


write.table(
    parameters,
    file=file.path(
        opt$outdir,
        "parameters.tsv"
    ),
    sep="\t",
    quote=FALSE,
    row.names=FALSE
)



# ---------------------------------------------------------
# Posterior probabilities
# ---------------------------------------------------------

message("[extract_viber_rds] Writing posterior.tsv")


posterior <- as.data.frame(
    fit$r_nk
)


colnames(posterior) <- names(
    fit$pi_k
)


posterior$mutation_index <- seq_len(
    nrow(posterior)
)


write.table(
    posterior,
    file=file.path(
        opt$outdir,
        "posterior.tsv"
    ),
    sep="\t",
    quote=FALSE,
    row.names=FALSE
)



# ---------------------------------------------------------
# ELBO
# ---------------------------------------------------------

if (!is.null(fit$ELBO)) {

    message("[extract_viber_rds] Writing elbo.tsv")


    elbo <- data.frame(

        iteration =
            seq_along(fit$ELBO),

        ELBO =
            fit$ELBO

    )


    write.table(
        elbo,
        file=file.path(
            opt$outdir,
            "elbo.tsv"
        ),
        sep="\t",
        quote=FALSE,
        row.names=FALSE
    )

} else {

    message(
        "[extract_viber_rds] No ELBO found"
    )

}



message("[extract_viber_rds] Finished.")
