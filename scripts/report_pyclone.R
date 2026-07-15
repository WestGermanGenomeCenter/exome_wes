#!/usr/bin/env Rscript

suppressPackageStartupMessages({

    library(data.table)
    library(ggplot2)
    library(scales)
    library(viridis)
    library(patchwork)
    library(pheatmap)
    library(RColorBrewer)

})


args <- commandArgs(trailingOnly = TRUE)

if(length(args) < 5){
    stop(
        paste(
            "Usage:",
            "plot_pyclone6_deep.R input.tsv results.tsv hdf5 pdf sample"
        )
    )
}


input_file  <- args[1]
result_file <- args[2]
hdf5_file   <- args[3]
pdf_file    <- args[4]
sample_name <- args[5]


theme_set(
    theme_bw(base_size=13)
)



############################################################
# Load data
############################################################


input <- fread(input_file)
res   <- fread(result_file)


dt <- merge(
    res,
    input,
    by=c("mutation_id","sample_id"),
    all.x=TRUE
)


dt[, depth := ref_counts + alt_counts]


dt[, chromosome := sub(":.*","",mutation_id)]

dt[, substitution :=
       sub(".*:","",mutation_id)]


dt[, substitution :=
       substr(substitution,1,3)]


dt[, confidence :=
       cluster_assignment_prob]



############################################################
# Summary statistics
############################################################


summary_table <- dt[,
    .(
        mutations=.N,
        mean_ccf=mean(cellular_prevalence),
        median_ccf=median(cellular_prevalence),
        mean_vaf=mean(variant_allele_frequency),
        mean_depth=mean(depth),
        mean_confidence=mean(confidence)
    ),
    by=cluster_id
]


setorder(summary_table,-mean_ccf)



############################################################
# PDF
############################################################


pdf(
    pdf_file,
    width=11,
    height=8.5
)



############################################################
# PAGE 1
# overview
############################################################


p1 <- ggplot(
    summary_table,
    aes(
        reorder(
            paste0("C",cluster_id),
            mean_ccf
        ),
        mean_ccf,
        fill=factor(cluster_id)
    )
)+
geom_col()+
coord_flip()+
scale_y_continuous(
    labels=percent
)+
scale_fill_viridis_d()+
labs(
    title=paste(
        "PyClone-VI clone overview",
        sample_name
    ),
    x="Cluster",
    y="Mean CCF"
)+
theme(
    legend.position="none"
)



grid <- paste(
    capture.output(summary_table),
    collapse="\n"
)


print(
    p1
)


############################################################
# PAGE 2
# CCF distributions
############################################################


p2 <- ggplot(
    dt,
    aes(
        factor(cluster_id),
        cellular_prevalence,
        fill=factor(cluster_id)
    )
)+
geom_violin(
    scale="width"
)+
geom_boxplot(
    width=.15,
    outlier.size=.2
)+
scale_y_continuous(
    labels=percent,
    limits=c(0,1)
)+
scale_fill_viridis_d()+
labs(
    title="Cancer cell fraction distributions",
    x="Cluster",
    y="CCF"
)+
theme(
    legend.position="none"
)


print(p2)



############################################################
# PAGE 3
# VAF distributions
############################################################


p3 <- ggplot(
    dt,
    aes(
        variant_allele_frequency,
        fill=factor(cluster_id)
    )
)+
geom_density(
    alpha=.4
)+
facet_wrap(~cluster_id)+
scale_x_continuous(
    labels=percent
)+
scale_fill_viridis_d()+
labs(
    title="VAF distributions by cluster",
    x="VAF",
    y="Density"
)


print(p3)



############################################################
# PAGE 4
# mutation spectra
############################################################


mutation_counts <- dt[,
    .N,
    by=.(cluster_id,substitution)
]


mutation_counts[
    ,
    fraction:=N/sum(N),
    by=cluster_id
]


p4 <- ggplot(
    mutation_counts,
    aes(
        substitution,
        fraction,
        fill=factor(cluster_id)
    )
)+
geom_col()+
facet_wrap(~cluster_id)+
scale_y_continuous(
    labels=percent
)+
scale_fill_viridis_d()+
labs(
    title="Cluster-specific mutation spectrum",
    x="Base substitution",
    y="Fraction"
)


print(p4)



############################################################
# PAGE 5
# confidence
############################################################


p5a <- ggplot(
    dt,
    aes(confidence)
)+
geom_histogram(
    bins=40,
    fill="steelblue"
)+
scale_x_continuous(
    limits=c(0,1)
)+
labs(
    title="Assignment confidence",
    x="P(cluster assignment)",
    y="Mutations"
)



p5b <- dt[,
    .(
        fraction_high_conf=
            mean(confidence>=0.9)
    ),
    by=cluster_id
]


p5 <- ggplot(
    p5b,
    aes(
        factor(cluster_id),
        fraction_high_conf,
        fill=factor(cluster_id)
    )
)+
geom_col()+
scale_y_continuous(
    labels=percent
)+
labs(
    title="% mutations with confidence >=0.9",
    x="Cluster",
    y=""
)+
theme(
    legend.position="none"
)



print(p5a+p5)



############################################################
# PAGE 6
# depth and VAF
############################################################


p6 <- ggplot(
    dt,
    aes(
        depth,
        variant_allele_frequency,
        colour=factor(cluster_id)
    )
)+
geom_point(
    alpha=.5
)+
scale_x_log10()+
scale_y_continuous(
    labels=percent
)+
scale_colour_viridis_d()+
labs(
    title="VAF versus sequencing depth",
    x="Read depth",
    y="VAF"
)


print(p6)



############################################################
# PAGE 7
# chromosome distribution
############################################################


chr <- dt[,
    .N,
    by=.(chromosome,cluster_id)
]


p7 <- ggplot(
    chr,
    aes(
        chromosome,
        N,
        fill=factor(cluster_id)
    )
)+
geom_col()+
scale_fill_viridis_d()+
theme(
    axis.text.x=
        element_text(
            angle=90,
            hjust=1
        )
)+
labs(
    title="Chromosomal mutation distribution",
    x="Chromosome",
    y="Mutations"
)



print(p7)



############################################################
# PAGE 8
# genome landscape
############################################################


dt[, position :=
    as.numeric(
        sub(
            ".*:(.*):.*",
            "\\1",
            mutation_id
        )
    )
]


p8 <- ggplot(
    dt,
    aes(
        position,
        variant_allele_frequency,
        colour=factor(cluster_id)
    )
)+
geom_point(
    size=.7,
    alpha=.6
)+
facet_wrap(~chromosome,
           scales="free_x")+
scale_y_continuous(
    labels=percent,
    limits=c(0,1)
)+
scale_colour_viridis_d()+
labs(
    title="Genome-wide VAF landscape",
    x="Position",
    y="VAF"
)



print(p8)



############################################################
# PAGE 9
# residual fit
############################################################


dt[, residual :=
       abs(
           variant_allele_frequency -
           cellular_prevalence/2
       )]


p9 <- ggplot(
    dt,
    aes(
        factor(cluster_id),
        residual,
        fill=factor(cluster_id)
    )
)+
geom_violin()+
scale_fill_viridis_d()+
labs(
    title="Within-cluster VAF residual",
    x="Cluster",
    y="|VAF - CCF/2|"
)+
theme(
    legend.position="none"
)


print(p9)



############################################################
# PAGE 10
# save summary table
############################################################


print(
    gridExtra::tableGrob(
        summary_table
    )
)



dev.off()



message(
    "Finished: ",
    pdf_file
)
# now copy all to fluse, make new run, test pyclone 6 reporting tool