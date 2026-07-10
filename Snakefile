import pandas as pd
from pathlib import Path

configfile: "config.yaml"

# ══════════════════════════════════════════════════════════════════════════════
# Samplesheet loading + validation
# ══════════════════════════════════════════════════════════════════════════════
def load_samplesheet(path):
    """Required cols: sample, type (tumor/normal), r1, r2. Optional: library."""
    ss = pd.read_csv(path, dtype=str).fillna("")

    req = {"sample", "type", "r1", "r2"}
    miss = req - set(ss.columns)
    if miss:
        raise ValueError(f"samplesheet.csv missing columns: {miss}")

    # check sample type values
    bad = set(ss["type"]) - {"tumor", "normal"}
    if bad:
        raise ValueError(f"Invalid type values (must be tumor/normal): {bad}")

    # check for duplicate FASTQ files
    fastqs = pd.concat([
        ss["r1"].rename("fastq"),
        ss["r2"].rename("fastq")
    ])

    dup_fastqs = fastqs[fastqs.duplicated(keep=False)]

    if not dup_fastqs.empty:
        raise ValueError(
            "FASTQ files appear multiple times in samplesheet:\n"
            + "\n".join(sorted(dup_fastqs.unique()))
        )

    counts = ss.groupby(["sample", "type"]).size().unstack(fill_value=0)

    for t in ("tumor", "normal"):
        if t not in counts.columns:
            raise ValueError(f"No '{t}' rows in samplesheet")

    bad_p = counts[(counts["tumor"] != 1) | (counts["normal"] != 1)]
    if not bad_p.empty:
        raise ValueError(
            f"Need exactly one tumor+normal per sample: {list(bad_p.index)}"
        )

    if "library" not in ss.columns:
        ss["library"] = ss["sample"] + "_" + ss["type"]

    ss = ss.set_index(["sample", "type"])

    return {
        s: {t: ss.loc[(s, t)] for t in ("tumor", "normal")}
        for s in ss.index.get_level_values("sample").unique()
    }



LOOKUP  = load_samplesheet(config["samplesheet"])
SAMPLES = list(LOOKUP.keys())

REF       = config["reference"]["fasta"]
DBSNP     = config["reference"]["dbsnp"]
INTERVALS = config["reference"]["intervals"]


REF_DIR = Path(REF).parent
# for deepsomatic: mount the ref dir

def get_r1(wc): return LOOKUP[wc.sample][wc.type]["r1"]
def get_r2(wc): return LOOKUP[wc.sample][wc.type]["r2"]

def get_rg(wc):
    lib = LOOKUP[wc.sample][wc.type].get("library", "")

    if not lib:
        lib = f"{wc.sample}_{wc.type}"

    return (
        f"@RG\\tID:{wc.sample}_{wc.type}"
        f"\\tSM:{wc.sample}_{wc.type}"
        f"\\tPL:ILLUMINA"
        f"\\tLB:{lib}"
        f"\\tPU:NA"
    )


wildcard_constraints:
    sample = "|".join(SAMPLES),
    type   = "tumor|normal",

OUTDIR = config["output_dir"]

# ══════════════════════════════════════════════════════════════════════════════
# Clustering tool selector
# ══════════════════════════════════════════════════════════════════════════════

CLUSTERING_TOOL = config.get("clustering_tool", "phylogic")

def cluster_outputs(sample):
    """Return final output paths for the active clustering tool."""
    if CLUSTERING_TOOL == "pyclone6":
        return [f"{OUTDIR}/{sample}/pyclone6/{sample}_pyclone6_results.tsv",
                f"{OUTDIR}/{sample}/pyclone6/{sample}_pyclone6_plot.pdf"]
    if CLUSTERING_TOOL == "viber":
        return [f"{OUTDIR}/{sample}/viber/{sample}_viber_clusters.tsv",
                f"{OUTDIR}/{sample}/viber/viber_report.pdf"]
    if CLUSTERING_TOOL == "orchard":
        return [f"{OUTDIR}/{sample}/orchard/{sample}_tree.pdf"]
    return [f"{OUTDIR}/{sample}/phylogic/{sample}.cluster_ccfs.txt",
            f"{OUTDIR}/{sample}/phylogic/{sample}.phylogic_report.html"]


# ══════════════════════════════════════════════════════════════════════════════
# Target rule
# ══════════════════════════════════════════════════════════════════════════════

rule all:
    input: # add kraken2 
        # QC
        f"{OUTDIR}/multiqc/multiqc_report.html",
        # Somatic calls
        expand("{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz", sample=SAMPLES,outdir=OUTDIR),
        expand("{outdir}/{sample}/deepsomatic/{sample}_report.pdf", sample=SAMPLES,outdir=OUTDIR),
        expand("{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",         sample=SAMPLES,outdir=OUTDIR),
        expand("{outdir}/{sample}/facets/{sample}_facets_qc.txt",            sample=SAMPLES,outdir=OUTDIR),
        expand("{outdir}/{sample}/cnaqc/{sample}_cnaqc_qc.txt",              sample=SAMPLES,outdir=OUTDIR),
        expand("{outdir}/{sample}/kraken/{sample}_report", sample=SAMPLES, outdir=OUTDIR)
            if config.get("kraken2", {}).get("kraken2_active", False) else [],
        # kraken conditional based on config.yaml setting
                # Clustering (tool-dependent)
        [f for s in SAMPLES for f in cluster_outputs(s)],


# ══════════════════════════════════════════════════════════════════════════════
# QC & Trimming
# ══════════════════════════════════════════════════════════════════════════════


rule trim_reads:
    # fastp: auto-detect adapters, quality filter, length filter.
    input:
        r1 = get_r1,
        r2 = get_r2,
    output:
        r1   = "{outdir}/{sample}/trimmed/{sample}_{type}_R1.fastq.gz",
        r2   = "{outdir}/{sample}/trimmed/{sample}_{type}_R2.fastq.gz",
        html = "{outdir}/{sample}/qc/fastp/{sample}_{type}_fastp.html",
        json = "{outdir}/{sample}/qc/fastp/{sample}_{type}_fastp.json",
    resources:
        threads  = lambda wildcards, attempt: attempt * 8,
        mem_gb   = lambda wildcards, attempt: 8 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message: "fastp trimming: {wildcards.sample} {wildcards.type}"
    log: "{outdir}/logs/{sample}/fastp_{type}.log" #     "{outdir}/{outdir}/logs/fastp/{sample}_{type}.log"
    conda: "envs/qc.yaml"
    params:
        min_length = config.get("trim",{}).get("min_length", 36),
        quality    = config.get("trim",{}).get("quality",    20),
    shell:
        """
        rm -f {output}
        fastp -i {input.r1} -I {input.r2} \
            -o {output.r1} -O {output.r2} \
            --html {output.html} --json {output.json} \
            --length_required {params.min_length} \
            --qualified_quality_phred {params.quality} \
            --detect_adapter_for_pe \
            --thread {resources.threads} 2>{log}
        """

rule fastqc_trimmed:
    # FastQC on trimmed FASTQs; verifies trimming didn't degrade quality.
    # Compare against fastqc_raw in MultiQC to confirm expected adapter removal.
    input:
        r1 = "{outdir}/{sample}/trimmed/{sample}_{type}_R1.fastq.gz",
        r2 = "{outdir}/{sample}/trimmed/{sample}_{type}_R2.fastq.gz",
    output:
        zip1  = "{outdir}/{sample}/qc/fastqc_trimmed/{sample}_{type}_R1_fastqc.zip",
        zip2  = "{outdir}/{sample}/qc/fastqc_trimmed/{sample}_{type}_R2_fastqc.zip",
        html1 = "{outdir}/{sample}/qc/fastqc_trimmed/{sample}_{type}_R1_fastqc.html",
        html2 = "{outdir}/{sample}/qc/fastqc_trimmed/{sample}_{type}_R2_fastqc.html",
    
    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 4 + attempt,
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "FastQC trimmed: {wildcards.sample} {wildcards.type}"
    log: "{outdir}/logs/{sample}/fastqc_trimmed_{type}.log"
    conda: "envs/qc.yaml"
    params:
        outdir = "{outdir}/{sample}/qc/fastqc_trimmed",
    shell: "fastqc -t {resources.threads} -o {params.outdir} {input.r1} {input.r2} 2>{log}"



# kraken2:
#        expand("{outdir}/{sample}/kraken/{sample}_report",              sample=SAMPLES,outdir=OUTDIR),
rule kraken2:
    input:
        r1   = "{outdir}/{sample}/trimmed/{sample}_tumor_R1.fastq.gz",
        r2   = "{outdir}/{sample}/trimmed/{sample}_normal_R2.fastq.gz",
    params:
        kraken2_db=config["kraken2"]["kraken2_database"],
        outfile=temp("{outdir}/{sample}/kraken/{sample}.kraken2"),
        outdir="{outdir}/{sample}/kraken/",
        summary="{outdir}/{sample}/kraken/{sample}.summary",
        read2_outfile = temp("{outdir}/{sample}/kraken/{sample}_h_.kraken2"),
        read2_report = "{outdir}/{sample}/kraken/{sample}_h_report"
    log:
        "{outdir}/logs/kraken/{sample}.kraken2.log"
    resources:
        threads=lambda wildcards, attempt: attempt * 2,
        time_hrs=lambda wildcards, attempt: attempt * 1,
        mem_gb=lambda wildcards, attempt: 178 + (attempt * 12)
    conda:
        "envs/kraken2.yaml"
    message: "kraken2: estimating organisms in the .fastq.gz files..."
    output:
        report_file="{outdir}/{sample}/kraken/{sample}_report"
    shell:
        """
        mkdir -p {params.outdir} 2>{log}
        kraken2 --use-names --db {params.kraken2_db} --threads {resources.threads} --gzip-compressed --quick --confidence 0.05 --report {output} {input.r1} >{params.outfile} 2>{log}
        kraken2 --use-names --db {params.kraken2_db} --threads {resources.threads} --gzip-compressed --quick --confidence 0.05 --report {params.read2_report} {input.r2} >{params.read2_outfile} 2>>{log}
        # this adds healty r2 to the cancer r1, so no contamination on either side should slip through
        chmod -f  ago+rwx -R {params.outdir} >> {log} 2>&1
        """

# ══════════════════════════════════════════════════════════════════════════════
# Alignment
# ══════════════════════════════════════════════════════════════════════════════

rule bwa_mem2_align:
    # Align paired-end reads; pipe to samtools sort. Raw BAM is marked temp.
    input:
        r1  = "{outdir}/{sample}/trimmed/{sample}_{type}_R1.fastq.gz",
        r2  = "{outdir}/{sample}/trimmed/{sample}_{type}_R2.fastq.gz",
#        ref = REF,
    output:
        sam = temp("{outdir}/{sample}/aligned/{sample}_{type}_raw.sam"),
    resources:
        threads  = lambda wildcards, attempt: attempt * 26,
        mem_gb   = lambda wildcards, attempt: 32 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 5,
    message: "BWA-MEM2 align: {wildcards.sample} {wildcards.type}"
    log: "{outdir}/logs/{sample}/bwa_{type}.log"
    conda: "envs/align.yaml"
    params:
        rg = get_rg,
        ref = config["reference"]["bwa_ref"],
        dir = "{outdir}/{sample}/aligned",
    shell:
        """
        rm -f {output.sam}
        mkdir -p {params.dir} 
        bwa-mem2 mem -t {resources.threads} -R '{params.rg}' {params.ref} {input.r1} {input.r2} >{output.sam} 2>>{log} 
        """


rule mark_duplicates:
    # samtools markdup: streaming version following canonical workflow:
    # collate → fixmate → position sort → markdup (no intermediate BAM files)
    input:
        sam = "{outdir}/{sample}/aligned/{sample}_{type}_raw.sam"
    output:
        bam     = "{outdir}/{sample}/aligned/{sample}_{type}_markdup.bam",
        bai     = "{outdir}/{sample}/aligned/{sample}_{type}_markdup.bam.bai",
        metrics = "{outdir}/{sample}/qc/samtools/{sample}_{type}_dupmetrics.txt",
        stats = "{outdir}/{sample}/qc/samtools/{sample}_{type}_stats.txt",
    resources:
        threads  = lambda wildcards, attempt: min(attempt * 16, 32),   # markdup is memory/I/O bound, not fully linear scaling
        mem_gb   = lambda wildcards, attempt: 16 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message:
        "samtools markdup : {wildcards.sample} {wildcards.type}"
    params:
        dir = "{outdir}/{sample}/aligned/",
        tmp_files="{outdir}/{sample}/aligned/{sample}_{type}*.tmp.bam",
    log:
        "{outdir}/logs/{sample}/markdup_{type}.log"
    conda:
        "envs/gatk.yaml"
    shell:
        """
        set -euo pipefail
        cd {params.dir} 2>> {log}
        rm -f {output.bam} rm -f {output.bai} # just to make sure on re-run of interrupted run that nothing gets concatenated. 
        rm -f  {params.tmp_files} # removing only tmp bam files of the current sample and type, so if the other one is still sorting its fine
        # now the actual samtools chain
        samtools collate -@ {resources.threads} -O -u {input.sam} | samtools fixmate -@ {resources.threads} -m -u - - | samtools sort -@ {resources.threads} -u - | samtools markdup -@ {resources.threads} - {output.bam}
        samtools index -@ {resources.threads} {output.bam}
        samtools flagstat {output.bam} > {output.metrics} 2>> {log}
        samtools stats {output.bam} > {output.stats} 2>> {log}
        """


rule mosdepth:
    # Per-target coverage QC over exome capture BED; quantized thresholds.
    input:
        bam = "{outdir}/{sample}/aligned/{sample}_{type}_markdup.bam",
        bed = INTERVALS,
    output:
        summary = "{outdir}/{sample}/qc/mosdepth/{sample}_{type}.mosdepth.summary.txt",
        regions = "{outdir}/{sample}/qc/mosdepth/{sample}_{type}.regions.bed.gz",
    
    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 8 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message: "mosdepth coverage: {wildcards.sample} {wildcards.type}"

    log: "{outdir}/logs/{sample}/mosdepth_{type}.log"

    conda: "envs/qc.yaml"

    params:
        prefix = "{outdir}/{sample}/qc/mosdepth/{sample}_{type}",

    shell:
        """
        mosdepth --threads {resources.threads} --by {input.bed} --quantize 0:10:20:50:100: {params.prefix} {input.bam} 2>{log}
        """


# ══════════════════════════════════════════════════════════════════════════════
# Somatic Calling
# ══════════════════════════════════════════════════════════════════════════════
#
# need to rebuild like this: singularity run -B /usr/lib/locale/:/usr/lib/locale/ deepsomatic_1.10.0.sif  run_deepsomatic works now as user
rule deepsomatic:
    # DeepSomatic v1.10.0: joint CNN on tumor+normal pileup images.
    # FILTER field: PASS=somatic, GERMLINE=in normal, RefCall=low-evidence.
    input:
        tumor  = "{outdir}/{sample}/aligned/{sample}_tumor_markdup.bam",
        normal = "{outdir}/{sample}/aligned/{sample}_normal_markdup.bam",
        ref    = REF,
        bed    = INTERVALS # so that its re-run if a new bed file is configured
    output:
        vcf  = "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        gvcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.g.vcf.gz",
    resources:
        threads  = lambda wildcards, attempt: attempt * 16,
        mem_gb   = lambda wildcards, attempt: 32 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 6,
    message: "DeepSomatic tumor-normal calling: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/deepsomatic.log"
    singularity:
        config.get("singularity",{}).get("deepsomatic","docker://google/deepsomatic:1.10.0")
    params:
        model_type = config.get("deepsomatic",{}).get("model_type","WES"),
        regions_arg = f"--regions {INTERVALS}" if config["deepsomatic"]["use_bed"] else "",
        tmpdir     = "{outdir}/{sample}/deepsomatic/tmp",
        extra      = config.get("deepsomatic",{}).get("extra_args",""),
        sif = config["singularity"]["deepsomatic"],
        bind_dir = REF_DIR
    shell:
        """
        mkdir -p {params.tmpdir}
        singularity run -B /usr/lib/locale/:/usr/lib/locale/ -B {params.bind_dir}:{params.bind_dir} {params.sif}  \
        run_deepsomatic \
            --model_type={params.model_type} --ref={input.ref} \
            --reads_tumor={input.tumor} --reads_normal={input.normal} \
            --output_vcf={output.vcf} {params.regions_arg} --output_gvcf={output.gvcf} \
            --num_shards={resources.threads} \
            --intermediate_results_dir={params.tmpdir} \
            {params.extra} > {log} 2>&1
        """


rule filter_deepsomatic:
    # Keep FILTER=PASS variants with AF >= min_vaf and DP >= min_depth.
    input:
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
    output:
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz",
        tbi = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz.tbi",
    
    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 2 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 8,
    
    message: "Filter DeepSomatic PASS: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/filter_deepsomatic.log"
    conda: "envs/bcftools.yaml"
    params:
        min_vaf   = config.get("somatic",{}).get("min_vaf",   0.05),
        min_depth = config.get("somatic",{}).get("min_depth", 10),
    shell:
        """
        bcftools index -t {input.vcf} -f 
        bcftools view -f PASS {input.vcf} \
        | bcftools filter \
            -i "FORMAT/VAF>={params.min_vaf} && FORMAT/DP>={params.min_depth}" \
        | bgzip -c > {output.vcf}
        tabix -p vcf {output.vcf} -f 
        """

rule somatic_qc:
    # bcftools stats + per-sample metrics via scripts/somatic_qc.sh (no awk).
    # Standalone equivalent:
    #   bash scripts/somatic_qc.sh sample_somatic_pass.vcf.gz \
    #       sample_somatic_qc.txt sample_somatic_bcfstats.txt SAMPLE_001
    input:
       # vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        #vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz",
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
    output:
        qc    = "{outdir}/{sample}/deepsomatic/{sample}_somatic_qc.txt",
        stats = "{outdir}/{sample}/deepsomatic/{sample}_somatic_bcfstats.txt",
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 4 + attempt,
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "Somatic variant QC: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/somatic_qc.log"
    conda: "envs/bcftools.yaml"
    shell:
        """
        bash scripts/somatic_qc.sh {input.vcf} {output.qc} {output.stats} \
            {wildcards.sample} > {log} 2>&1
        """



rule deepsomatic_qc:
     input:
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
    output:
        qc    = "{outdir}/{sample}/deepsomatic/{sample}_report.pdf",
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 4 + attempt,
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "Deepsomatic report: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/deepsomatic_report.log"
    conda: "envs/plot_deepsomatic.yaml"
    shell:
        """
        python3 scripts/deepsomatic_report.py --input {input.vcf} --output {output.qc} > {log} 2>&1
        """


# ══════════════════════════════════════════════════════════════════════════════
# Copy Number
# ══════════════════════════════════════════════════════════════════════════════

rule snp_pileup:
    # Allele counts at dbSNP positions for FACETS. Normal must be passed first.
    input:
        normal = "{outdir}/{sample}/aligned/{sample}_normal_markdup.bam",
        tumor  = "{outdir}/{sample}/aligned/{sample}_tumor_markdup.bam",
        dbsnp  = DBSNP,
    output:
        pileup = "{outdir}/{sample}/facets/{sample}_pileup.csv.gz",
    
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 8 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 3,
    message: "snp-pileup for FACETS: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/snp_pileup.log"
    conda: "envs/facets.yaml"
    params:
        min_bq = config.get("facets",{}).get("min_base_quality", 25),
        min_mq = config.get("facets",{}).get("min_map_quality",  15),
        min_dn = config.get("facets",{}).get("min_depth_normal", 25),
    shell:
        """
        snp-pileup -g \
            -q {params.min_mq} -Q {params.min_bq} -P 100 \
            -r {params.min_dn},0 \
            {input.dbsnp} {output.pileup} \
            {input.normal} {input.tumor} 2>{log}
        """

rule run_facets:
    # FACETS: GC-correct → CBS segment → EM allele-specific CN fit.
    # Purity and CN segments flow into cnaqc, phylogic_cluster, pyclone6.
    input:
        pileup = "{outdir}/{sample}/facets/{sample}_pileup.csv.gz",
    output:
        rds = "{outdir}/{sample}/facets/{sample}_facets.rds",
        seg = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
        pdf = "{outdir}/{sample}/facets/{sample}_cnv.pdf",
        qc  = "{outdir}/{sample}/facets/{sample}_facets_qc.txt",
    
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 16 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message: "FACETS copy number: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/facets.log"
    conda: "envs/facets.yaml"
    params:
        cval     = config.get("facets",{}).get("cval",    150),
        min_nhet = config.get("facets",{}).get("min_nhet", 25),
        ndepth   = config.get("facets",{}).get("ndepth",   35),
        snp_nbhd = config.get("facets",{}).get("snp_nbhd",250),
        genome   = config.get("facets",{}).get("genome", "hg38"),
        sname    = "{sample}",
    shell:
        """
        Rscript scripts/run_facets.R \
            {input.pileup} {output.rds} {output.seg} {output.pdf} {output.qc} \
            {params.sname} {params.cval} {params.min_nhet} \
            {params.ndepth} {params.snp_nbhd} {params.genome} 2>{log}
        """

rule cnaqc:
    # CNAqc: checks somatic VAF peaks match expected CN peaks given purity.
    # Standalone equivalent:
    #   Rscript --vanilla scripts/run_cnaqc.R \
    #       sample_somatic_pass.vcf.gz sample_cnv_segments.tsv sample_facets.rds \
    #       sample_cnaqc_qc.txt sample_cnaqc_plot.pdf sample_cnaqc.rds \
    #       SAMPLE_001 0.05
    input:
       # vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz",
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        seg = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
        rds = "{outdir}/{sample}/facets/{sample}_facets.rds",
    output:
        qc   = "{outdir}/{sample}/cnaqc/{sample}_cnaqc_qc.txt",
        plot = "{outdir}/{sample}/cnaqc/{sample}_cnaqc_plot.pdf",
        rds  = "{outdir}/{sample}/cnaqc/{sample}_cnaqc.rds",
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 12 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "CNAqc CN validation: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/cnaqc.log"
    conda: "envs/cnaqc.yaml"
    params:
        purity_tol = config.get("cnaqc", {}).get("purity_tolerance", 0.05),
    shell:
        """
        Rscript --vanilla scripts/run_cnaqc.R \
            {input.vcf} {input.seg} {input.rds} \
            {output.qc} {output.plot} {output.rds} \
            {wildcards.sample} {params.purity_tol} \
            > {log} 2>&1
        """





# ══════════════════════════════════════════════════════════════════════════════
# PhylogicNDT — three rules. All shell-only; no Snakemake script: directive.
# Equivalent commands can be run directly on the CLI for debugging — just
# substitute the {wildcards}/{input}/{output} placeholders with real paths.
# ══════════════════════════════════════════════════════════════════════════════

rule phylogic_prep:
    # Builds PhylogicNDT MAF + writes purity to a plain-text file.
    # Standalone equivalent:
    #   Rscript --vanilla scripts/build_phylogic_input.R \
    #       sample_somatic_pass.vcf.gz sample_cnv_segments.tsv \
    #       sample_facets.rds sample.maf sample_purity.txt SAMPLE_001
    input:
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        #vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz",
        seg = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
        rds = "{outdir}/{sample}/facets/{sample}_facets.rds",
    output:
        maf    = "{outdir}/{sample}/phylogic/{sample}_phylogic_input.maf",
        purity = "{outdir}/{sample}/phylogic/{sample}_purity.txt",
        seg     = "{outdir}/{sample}/phylogic/{sample}_phylogic_segments.seg",
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 8 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message: "PhylogicNDT prep (MAF + purity): {wildcards.sample}"
    log: "{outdir}/logs/{sample}/phylogic_prep.log"
    conda: "envs/cnaqc.yaml"
    shell:
        """
        Rscript --vanilla scripts/build_phylogic_input2.R \
            {input.vcf} {input.seg} {input.rds} \
            {output.maf} {output.seg} {output.purity} {wildcards.sample} \
            > {log} 2>&1
        """





rule phylogic_prepare_sif:
    input:
        maf    = "{outdir}/{sample}/phylogic/{sample}_phylogic_input.maf",
        seg    = "{outdir}/{sample}/phylogic/{sample}_phylogic_segments.seg",
        purity = "{outdir}/{sample}/phylogic/{sample}_purity.txt",
    output:
        sif = "{outdir}/{sample}/phylogic/{sample}.sif",
    resources:
        threads  = lambda wildcards, attempt: attempt * 1,
        mem_gb   = lambda wildcards, attempt: 1 + (attempt * 1),
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "Creating PhylogicNDT SIF: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/phylogic_prepare_sif.log"
    shell:
        """
        purity=$(cat {input.purity})
        printf "sample_id\tmaf_fn\tseg_fn\tpurity\ttimepoint\n" > {output.sif}
        printf "{wildcards.sample}\t{input.maf}\t{input.seg}\t${{purity}}\t1\n" >> {output.sif}
        """

rule phylogic_cluster:
    # PhylogicNDT Cluster via the inline -s sample-info string:
    #   sample_id:maf:seg:purity
    input:
        sif    = "{outdir}/{sample}/phylogic/{sample}.sif",
        maf    = "{outdir}/{sample}/phylogic/{sample}_phylogic_input.maf",
        #seg    = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
        seg     = "{outdir}/{sample}/phylogic/{sample}_phylogic_segments.seg",
    output:
        posteriors = "{outdir}/{sample}/phylogic/{sample}.cluster_ccfs.txt",
        mut_ccf     = "{outdir}/{sample}/phylogic/{sample}.mut_ccfs.txt",

    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 16 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 4,
    message: "PhylogicNDT Cluster: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/phylogic_cluster.log"
    conda: "envs/phylogic.yaml"
    params:
        phylogicndt = config["phylogic"]["phylogicndt_dir"] + "/PhylogicNDT.py",
        n_iter      = config.get("phylogic", {}).get("n_iter", 1000),
        outdir      = "{outdir}/{sample}/phylogic",
    shell:
        """
        cd {params.outdir}
        python {params.phylogicndt} Cluster \
            -i {wildcards.sample} \
            -sif {input.sif} \
            -ni {params.n_iter} \
            --seed 42 \
            > {log} 2>&1
        """
# removed             --maf_input_type calc_ccf \ for testing, new build script makes the ccf in the prep stage
# added timepoint: :1 at argument -s at the end, only one timepoint available

rule phylogic_build_tree:
    # PhylogicNDT BuildTree, same inline -s syntax as Cluster.
    # Standalone equivalent (after Cluster has run in the same directory):
    #   python PhylogicNDT.py BuildTree -i SAMPLE_001 \
    #       -s SAMPLE_001:sample.maf:sample_seg.tsv:0.65 --seed 42
    input:
        sif    = "{outdir}/{sample}/phylogic/{sample}.sif",
        posteriors = "{outdir}/{sample}/phylogic/{sample}.cluster_ccfs.txt",
        mut_ccf     = "{outdir}/{sample}/phylogic/{sample}.mut_ccfs.txt",
    output:
        html  = "{outdir}/{sample}/phylogic/{sample}.phylogic_report.html",
    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 16 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message: "PhylogicNDT BuildTree: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/phylogic_build_tree.log"
    conda: "envs/phylogic.yaml"
    params:
        phylogicndt = config["phylogic"]["phylogicndt_dir"] + "/PhylogicNDT.py",
        outdir      = "{outdir}/{sample}/phylogic",
    shell:
        """
        cd {params.outdir}
        python {params.phylogicndt} BuildTree \
            -i {wildcards.sample} \
            -sif {input.sif} \
            --cluster_ccf {input.posteriors} \
            --mutation_ccf {input.mut_ccf} \
            --seed 42 \
            > {log} 2>&1

        """


# ══════════════════════════════════════════════════════════════════════════════
# PyClone6 — alternative clustering branch. Two rules, same style.
# Active when clustering_tool: pyclone6 in config.yaml.
# ══════════════════════════════════════════════════════════════════════════════

rule pyclone6_prep:
    # Builds the PyClone6 input TSV (mutation_id, ref/alt counts, CN, purity).
    # Standalone equivalent:
    #   Rscript --vanilla scripts/build_pyclone6_input.R \
    #       sample_somatic_pass.vcf.gz sample_cnv_segments.tsv \
    #       sample_facets.rds sample_pyclone6_input.tsv SAMPLE_001
    input:
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        seg = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
        rds = "{outdir}/{sample}/facets/{sample}_facets.rds",
    output:
        tsv = "{outdir}/{sample}/pyclone6/{sample}_pyclone6_input.tsv",
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 8 + (attempt * 4),
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "PyClone6 prep: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/pyclone6_prep.log"
    conda: "envs/cnaqc.yaml"
    shell:
        """
        Rscript --vanilla scripts/build_pyclone6_input.R \
            {input.vcf} {input.seg} {input.rds} \
            {output.tsv} {wildcards.sample} \
            > {log} 2>&1
        """


rule pyclone6:
    # PyClone6 CLI: run_analysis → build_results_file → plot_pyclone6.R.
    # Standalone equivalent:
    #   pyclone6 run_analysis --in_files sample_pyclone6_input.tsv \
    #       --out_dir outdir --num_clusters 10 --num_restarts 10 \
    #       --num_iters 10000 --density beta-binomial
    #   pyclone6 build_results_file --in_dir outdir --out_file results.tsv
    #   Rscript --vanilla scripts/plot_pyclone6.R results.tsv plot.pdf SAMPLE_001
    input:
        tsv = "{outdir}/{sample}/pyclone6/{sample}_pyclone6_input.tsv",
    output:
        results = "{outdir}/{sample}/pyclone6/{sample}_pyclone6_results.tsv",
        plot    = "{outdir}/{sample}/pyclone6/{sample}_pyclone6_plot.pdf",
    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 16 + (attempt * 8),
        time_hrs = lambda wildcards, attempt: attempt * 4,
    message: "PyClone6 clustering: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/pyclone6.log"
    conda: "envs/pyclone6.yaml"
    params:
        n_clusters = config.get("pyclone6", {}).get("n_clusters",  10),
        n_restarts = config.get("pyclone6", {}).get("n_restarts",  10),
        num_iters  = config.get("pyclone6", {}).get("num_iters", 10000),
        density    = config.get("pyclone6", {}).get("density", "beta-binomial"),
        outdir     = "{outdir}/{sample}/pyclone6.hdf5",
        results_tmp = "{outdir}/{sample}/pyclone6/{sample}_pyclone6_results_tmp.tsv",
    shell:
        """
        pyclone-vi fit \
            -i {input.tsv} \
            -o {params.outdir} \
            -c {params.n_clusters} \
            -r {params.n_restarts} \
            --max-iters {params.num_iters} --num-threads {resources.threads} \
            --density {params.density} \
            > {log} 2>&1

        pyclone-vi write-results-file \
            -i {params.outdir} \
            -o {params.results_tmp} \
            >> {log} 2>&1


        python3 scripts/add_vaf_to_pyclone6_results.py \
            {input.tsv} {params.results_tmp} {output.results} \
            >> {log} 2>&1



        Rscript --vanilla scripts/plot_pyclone6.R \
            {output.results} {output.plot} {wildcards.sample} \
            >> {log} 2>&1
        """


# ══════════════════════════════════════════════════════════════════════════════
# Orchard — orthogonal tumor phylogeny via stochastic combinatorial search.
# Requires PyClone6 cluster assignments as input (pyclone6 must run first).
# Uses identical .ssm/.params.json format as Pairtree.
# Visualisation reuses Pairtree's plottree (both repos must be cloned).
# Active when clustering_tool: orchard in config.yaml.
# ══════════════════════════════════════════════════════════════════════════════

rule orchard_prep:
    # Builds .ssm (read counts + CN-corrected var_read_prob) and
    # .params.json (PyClone6 cluster assignments) for Orchard input.
    input:
        vcf     = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        seg     = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
        pyclone = "{outdir}/{sample}/pyclone6/{sample}_pyclone6_results.tsv",
    output:
        ssm    = "{outdir}/{sample}/orchard/{sample}.ssm",
        params = "{outdir}/{sample}/orchard/{sample}.params.json",
    resources:
        threads  = lambda wildcards, attempt: 1,
        mem_gb   = lambda wildcards, attempt: 8 + attempt * 4,
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "Orchard input prep: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/orchard_prep.log"
    conda: "envs/cnaqc.yaml"
    shell:
        """
        Rscript --vanilla scripts/build_orchard_input.R \
            {input.vcf} {input.seg} {input.pyclone} \
            {output.ssm} {output.params} {wildcards.sample} \
            > {log} 2>&1
        """

rule orchard_run:
    # Orchard stochastic combinatorial search → ranked clone trees (.npz).
    # plottree (from Pairtree) produces the interactive HTML report.
    # Key params: -k beam width, -f branching factor, -n parallel instances.
    # For WES with few clusters (typically <15): default k=10, f=100 is fine.
    input:
        ssm    = "{outdir}/{sample}/orchard/{sample}.ssm",
        params = "{outdir}/{sample}/orchard/{sample}.params.json",
    output:
        npz  = "{outdir}/{sample}/orchard/{sample}.orchard.npz",
        plot   = "{outdir}/{sample}/orchard/{sample}_tree.pdf", # Changed to PDF    
    resources:
        threads  = lambda wildcards, attempt: attempt * 4,
        mem_gb   = lambda wildcards, attempt: 16 + attempt * 8,
        time_hrs = lambda wildcards, attempt: attempt * 2,
    message: "Orchard tree inference: {wildcards.sample}"
    log: "{outdir}/logs/{sample}/orchard_run.log"
    conda: "envs/orchard.yaml"
    params:
        orchard      = config["orchard"]["orchard_dir"] + "/bin/orchard",
        plottree     = config["orchard"]["pairtree_dir"] + "/bin/plottree",
        beam_width   = config.get("orchard", {}).get("beam_width",   10),
        branching    = config.get("orchard", {}).get("branching",   100),
        n_parallel   = config.get("orchard", {}).get("n_parallel",    4),
        outdir       = "{outdir}/{sample}/orchard",
    shell:
        """
        python3 {params.orchard} \
            {input.ssm} \
            {input.params} \
            {output.npz} \
            -k {params.beam_width} \
            -f {params.branching} \
            -n {params.n_parallel} \
            > {log} 2>&1


        python3 scripts/plot_orchard_tree.py {output.npz} {output.plot} >> {log} 2>&1
        """

        
# ══════════════════════════════════════════════════════════════════════════════
# Experimental and might be discarded- viber
# ══════════════════════════════════════════════════════════════════════════════




rule viber_prep:
    input:
        vcf = "{outdir}/{sample}/deepsomatic/{sample}_somatic_pass.vcf.gz" if config["somatic"]["filter"] else "{outdir}/{sample}/deepsomatic/{sample}_somatic_raw.vcf.gz",
        seg = "{outdir}/{sample}/facets/{sample}_cnv_segments.tsv",
    output:
        tsv = "{outdir}/{sample}/viber/{sample}_viber_input.tsv"

    resources:
        threads = lambda wildcards, attempt: attempt * 2,
        mem_gb = lambda wildcards, attempt: 4 + attempt * 2,
        time_hrs = lambda wildcards, attempt: attempt * 1

    message:
        "VIBER input preparation: {wildcards.sample}"

    log:
        "{outdir}/logs/{sample}/viber_prep.log"

    conda:
        "envs/viber.yaml"

    shell:
        """
        Rscript --vanilla scripts/build_viber_input.R \
            {input.vcf} {input.seg} \
            {output.tsv} \
            > {log} 2>&1
        """



rule viber:
    input:
        tsv = "{outdir}/{sample}/viber/{sample}_viber_input.tsv"
    output:
        clusters = "{outdir}/{sample}/viber/{sample}_viber_clusters.tsv",
        rds ="{outdir}/{sample}/viber/{sample}_viber_fit.rds"
    resources:
        threads = lambda wildcards, attempt: attempt * 2,
        mem_gb = lambda wildcards, attempt: 18 + attempt * 4,
        time_hrs = lambda wildcards, attempt: attempt * 2
    message:
        "VIBER clustering: {wildcards.sample}"
    log:
        "{outdir}/logs/{sample}/viber.log"
    conda:
        "envs/viber.yaml"
    params:
        prefix = "{outdir}/{sample}/viber/{sample}",
        dir = "{outdir}/{sample}/viber/"
    shell:
        """
        Rscript --vanilla scripts/run_viber.R \
            {input.tsv} \
            {params.prefix} {params.dir} \
            > {log} 2>&1
        """


# now plot this all

rule extract_viber:
    input:
        tsv = "{outdir}/{sample}/viber/{sample}_viber_input.tsv",
        rds ="{outdir}/{sample}/viber/{sample}_viber_fit.rds",
        clusters = "{outdir}/{sample}/viber/{sample}_viber_clusters.tsv"
    output:
        posterios = "{outdir}/{sample}/viber/posterior.tsv"
    resources:
        threads = lambda wildcards, attempt: attempt * 2,
        mem_gb = lambda wildcards, attempt: 2 + attempt * 4,
        time_hrs = lambda wildcards, attempt: attempt * 1
    message:
        "extracting VIBER clustering rds: {wildcards.sample}"
    log:
        "{outdir}/logs/{sample}/viber_extract.log"
    conda:
        "envs/viber.yaml"
    params:
        dir = "{outdir}/{sample}/viber/"
    shell:
        """
        Rscript  --vanilla scripts/extract_viber.R --rds {input.rds} --outdir {params.dir} --input {input.tsv}  > {log} 2>&1
        """

rule report_viber:
    input:
        tsv = "{outdir}/{sample}/viber/{sample}_viber_input.tsv",
        rds ="{outdir}/{sample}/viber/{sample}_viber_fit.rds",
        clusters = "{outdir}/{sample}/viber/{sample}_viber_clusters.tsv"
    output:
        viber_pdf = "{outdir}/{sample}/viber/viber_report.pdf"
    resources:
        threads = lambda wildcards, attempt: attempt * 2,
        mem_gb = lambda wildcards, attempt: 2 + attempt * 4,
        time_hrs = lambda wildcards, attempt: attempt * 1
    message:
        "extracting VIBER clustering rds: {wildcards.sample}"
    log:
        "{outdir}/logs/{sample}/viber_extract.log"
    conda:
        "envs/report_viber.yaml"
    params:
        dir = "{outdir}/{sample}/viber/"
        sample_name = "{sample}"
    shell:
        """
        python scripts/viber_report.py --dir {params.dir} --sample {params.sample_name} --output {output.viber_pdf} > {log} 2>&1
        """


# ══════════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════════

rule multiqc:
    # Aggregates: FastQC (raw + trimmed), fastp, GATK MarkDuplicates,
    # mosdepth, bcftools stats. One HTML report per pipeline run.
    input:
#        fqc_raw   = expand("{outdir}/{sample}/qc/fastqc_raw/{sample}_{type}_R1_fastqc.zip",
#                           sample=SAMPLES, type=["tumor","normal"]),
        fqc_trim  = expand("{outdir}/{sample}/qc/fastqc_trimmed/{sample}_{type}_R1_fastqc.zip",
                           sample=SAMPLES, type=["tumor","normal"],outdir=OUTDIR),
        fastp     = expand("{outdir}/{sample}/qc/fastp/{sample}_{type}_fastp.json",
                           sample=SAMPLES, type=["tumor","normal"],outdir=OUTDIR),
#        picard    = expand("{outdir}/{sample}/qc/picard/{sample}_{type}_dupmetrics.txt",
#                           sample=SAMPLES, type=["tumor","normal"],outdir=OUTDIR),
        mosdepth  = expand("{outdir}/{sample}/qc/mosdepth/{sample}_{type}.mosdepth.summary.txt",
                           sample=SAMPLES, type=["tumor","normal"],outdir=OUTDIR),
#        bcfstats  = expand("{outdir}/{sample}/deepsomatic/{sample}_somatic_bcfstats.txt",
#                           sample=SAMPLES,outdir=OUTDIR), # enable later again once without-filter works
    output:
        html = "{outdir}/multiqc/multiqc_report.html",
    
    resources:
        threads  = lambda wildcards, attempt: attempt * 2,
        mem_gb   = lambda wildcards, attempt: 8 + attempt,
        time_hrs = lambda wildcards, attempt: attempt * 1,
    message: "MultiQC aggregation"
    log: "{outdir}/logs/multiqc.log"
    conda: "envs/qc.yaml"
    params:
        outdir = OUTDIR
    shell:
        """
        multiqc {params.outdir}  -o {params.outdir}/multiqc  --filename multiqc_report.html --force >{log} 2>&1
        """
