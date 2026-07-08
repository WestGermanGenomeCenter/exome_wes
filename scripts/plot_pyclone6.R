#!/usr/bin/env Rscript
# scripts/plot_pyclone6.R  args: results_tsv output_pdf sample_name
suppressPackageStartupMessages({
  library(data.table); library(ggplot2); library(scales)
})

args  <- commandArgs(trailingOnly = TRUE)
dt    <- fread(args[1])
sname <- args[3]

clust <- dt[, .(
  n        = .N,
  mean_ccf = mean(cellular_prevalence, na.rm = TRUE),
  sd_ccf   = sd(cellular_prevalence,   na.rm = TRUE)
), by = cluster_id]
setorder(clust, -mean_ccf)
clust[, lbl := paste0("C", cluster_id, " (n=", n, ")")]
clust[, lbl := factor(lbl, levels = rev(lbl))]

p1 <- ggplot(clust, aes(mean_ccf, lbl, colour = lbl)) +
  geom_segment(aes(xend = 0, yend = lbl), colour = "grey70", linewidth = 0.6) +
  geom_errorbarh(aes(xmin = pmax(mean_ccf - sd_ccf, 0),
                     xmax = pmin(mean_ccf + sd_ccf, 1)),
                 height = 0.3, linewidth = 0.7) +
  geom_point(size = 4) +
  scale_x_continuous(limits = c(0, 1), labels = percent_format()) +
  scale_colour_viridis_d(option = "plasma", end = 0.85, guide = "none") +
  labs(title = paste0("Clonal Architecture — ", sname),
       x = "Cancer Cell Fraction (CCF)", y = "Clone") +
  theme_bw(base_size = 13) +
  theme(plot.title = element_text(face = "bold"), panel.grid.minor = element_blank())

p2 <- ggplot(dt, aes(variant_allele_frequency, cellular_prevalence,
                     colour = factor(cluster_id))) +
  geom_point(alpha = 0.5, size = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey50") +
  scale_x_continuous(limits = c(0, 1), labels = percent_format()) +
  scale_y_continuous(limits = c(0, 1), labels = percent_format()) +
  scale_colour_viridis_d(option = "plasma", end = 0.85, name = "Cluster") +
  labs(title = "VAF vs CCF (diagnostic)",
       x = "Variant Allele Frequency", y = "Cancer Cell Fraction") +
  theme_bw(base_size = 13)

pdf(args[2], width = 10, height = 6)
print(p1)
print(p2)
dev.off()
