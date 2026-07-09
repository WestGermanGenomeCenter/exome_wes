#!/usr/bin/env Rscript --vanilla
# scripts/plot_orchard_tree.R

# ── Setup and Logging ─────────────────────────────────────────────────────────

log_msg <- function(msg, level = "INFO") {
  cat(sprintf("[%s] %s: %s\n", Sys.time(), level, msg), sep = "")
}

log_msg("Starting plot_orchard_tree.R")

suppressPackageStartupMessages({
    library(jsonlite)
    library(reticulate)
    library(ape)
})

# Check required libraries
required_libs <- c("jsonlite", "reticulate", "ape")

missing_libs <- required_libs[
    !sapply(
        required_libs,
        require,
        character.only = TRUE,
        quietly = TRUE
    )
]

if (length(missing_libs) > 0) {
    stop(paste(
        "Missing required R libraries:",
        paste(missing_libs, collapse = ", "),
        "\nPlease check your conda env.yaml"
    ))
}


# ── Arguments ────────────────────────────────────────────────────────────────

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
    stop(
        "Usage: plot_orchard_tree.R <ssm> <params_json> <npz> <out_pdf>"
    )
}

ssm_file    <- args[1]
params_file <- args[2]
npz_file    <- args[3]
out_pdf     <- args[4]


# ── Sanity Checks ───────────────────────────────────────────────────────────

for (f in c(ssm_file, params_file, npz_file)) {
    if (!file.exists(f)) {
        stop(sprintf("Input file not found: %s", f))
    }
}

log_msg("All input files found.")


# ── Python / NumPy Integration ──────────────────────────────────────────────

log_msg("Configuring reticulate Python path...")

py_path <- Sys.which("python3")

if (py_path == "") {
    stop("Could not find python3 in the current PATH.")
}

log_msg(sprintf("Using Python executable: %s", py_path))


tryCatch({

    reticulate::use_python(py_path, required = TRUE)

    np <- reticulate::import("numpy")

    log_msg("Successfully imported numpy via reticulate.")

}, error = function(e) {

    stop(sprintf(
        "Failed to initialize Python/Numpy: %s",
        e$message
    ))

})


# ── Load Orchard NPZ ────────────────────────────────────────────────────────

log_msg("Loading Orchard .npz data...")

tryCatch({

    data <- np$load(
        npz_file,
        allow_pickle = TRUE
    )

    # FIX: Convert Python KeysView to list, then to R vector
    keys <- reticulate::py_to_r(list(data$keys()))
    keys <- as.character(unlist(keys))

    log_msg(sprintf(
        "NPZ keys found: %s",
        paste(keys, collapse = ", ")
    ))

    if (!"newick" %in% keys) {
        stop(
            "The Orchard NPZ file does not contain a 'newick' tree."
        )
    }

    # Orchard ranks candidate trees.
    # The first Newick string is the top-ranked solution.

    newick_array <- data[["newick"]]
    newicks <- reticulate::py_to_r(newick_array$tolist())
    best_tree <- newicks[1]

    log_msg(sprintf(
        "Selected Orchard tree: %s",
        best_tree
    ))

    tree <- ape::read.tree(
        text = best_tree
    )

}, error = function(e) {

    stop(sprintf(
        "Error reading Orchard tree: %s",
        e$message
    ))

})



# ── Load params (optional metadata check) ───────────────────────────────────

log_msg("Loading params.json...")

tryCatch({

    params <- jsonlite::fromJSON(params_file)

    if (!"clusters" %in% names(params)) {
        warning(
            "params.json does not contain 'clusters' key."
        )
    }

}, error = function(e) {

    stop(sprintf(
        "Error parsing params.json: %s",
        e$message
    ))

})


# ── Plot Tree ───────────────────────────────────────────────────────────────

log_msg(sprintf(
    "Generating PDF plot: %s",
    out_pdf
))


tryCatch({

    pdf(
        out_pdf,
        width = 8,
        height = 8
    )


    plot(
        tree,
        cex = 1,
        main = "Orchard phylogenetic tree"
    )


    dev.off()


    log_msg("PDF successfully saved.")


}, error = function(e) {

    if (dev.cur() > 1) {
        dev.off()
    }

    stop(sprintf(
        "Plotting failed: %s",
        e$message
    ))

})


log_msg("Process completed successfully.")