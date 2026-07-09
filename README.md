# Paired Tumour-Normal Whole-Exome Sequencing Somatic Pipeline

A Snakemake pipeline for detecting somatic mutations, copy number alterations, and tumour clonal architecture from paired tumour-normal WES data. Developed at the Biomedical Research Facility (bmfz/GTL), Heinrich-Heine-Universität Düsseldorf.

---

## What Does This Pipeline Do, and Why?

When a cancer develops, it doesn't arise from a single mutation — it accumulates genetic changes over time. Some of these changes are present in all tumour cells (clonal), while others arose later in subsets of cells (subclonal). Understanding this architecture — which mutations came first, which are expanding, and how many distinct subclones exist — is critical for predicting treatment resistance and disease progression.

This pipeline takes raw sequencing data from a tumour biopsy and a matched normal blood sample and reconstructs that evolutionary history in several steps:

1. **Clean the raw reads** — remove low-quality bases and sequencing adapters
2. **Align to the human genome** — find where each read came from
3. **Detect somatic mutations** — identify changes present in the tumour but not the normal
4. **Estimate copy number** — determine how many copies of each chromosomal region the tumour has
5. **Validate copy number** — check that the copy number calls are consistent with the mutation data
6. **Reconstruct clonal evolution** — group mutations by when they arose and build a phylogenetic tree of the tumour's subclones

---

## Biological Background

### Why Paired Tumour-Normal?

Every person carries millions of germline variants — differences from the reference genome that are present in every cell of their body. To find somatic mutations (those that arose specifically in the tumour), we sequence both tumour and matched normal (usually blood) DNA and subtract the germline signal. A variant in the tumour but not the normal is somatic; a variant in both is germline.

### What is Variant Allele Frequency (VAF)?

When we sequence a tumour, we get reads from a mixture of tumour cells and contaminating normal stromal cells. For a heterozygous somatic mutation (one copy mutated, one copy wild-type) in a pure diploid tumour, we expect to see the variant in ~50% of reads (VAF = 0.5). But:

- If tumour purity is 70% (30% normal contamination), VAF drops to ~0.35
- If the mutation is on a region with copy number gain (e.g. 3 copies), the expected VAF changes again
- If the mutation is subclonal (only in 50% of tumour cells), the expected VAF halves again

This is why raw VAF is not directly interpretable without knowing purity and copy number.

### What is Cancer Cell Fraction (CCF)?

CCF is the proportion of cancer cells that carry a given mutation, corrected for purity and local copy number. A CCF of 1.0 means the mutation is clonal — present in all tumour cells and therefore arose early. A CCF of 0.3 means only 30% of tumour cells carry it — it is subclonal and arose later in a subset.

CCF is estimated as a probability distribution rather than a point estimate, because sequencing is noisy. The uncertainty is larger for shallow coverage and smaller for deep coverage.

---

## Pipeline Steps

### Step 1 — Read Trimming (`trim_reads`)

**Tool:** fastp

Raw sequencing reads contain adapter sequences (synthetic DNA added during library preparation) that must be removed before alignment. fastp also trims low-quality bases from read ends and discards very short reads. This prevents spurious alignments and false-positive variant calls.

### Step 2 — Quality Control (`fastqc_trimmed`, `multiqc`)

**Tools:** FastQC, MultiQC

FastQC checks the trimmed reads for quality metrics (per-base quality scores, GC content, duplication rate, k-mer content). MultiQC aggregates all QC results from all steps into a single HTML report for easy review.

### Step 3 — Alignment (`bwa_mem2_align`)

**Tool:** BWA-MEM2

Each read is aligned to the human reference genome (hg38) to determine its genomic origin. BWA-MEM2 is a fast, accurate aligner for short paired-end reads. The output is a SAM file containing each read's position, mapping quality, and any mismatches relative to the reference.

### Step 4 — Duplicate Marking (`mark_duplicates`)

**Tool:** samtools (collate → fixmate → sort → markdup)

During library preparation, individual DNA molecules are amplified by PCR. This creates multiple identical copies of the same original molecule — PCR duplicates — which are not independent observations. If left in, they artificially inflate variant counts and confidence. This step identifies read pairs with identical start and end positions and marks them as duplicates so they are excluded from downstream analysis.

The pipeline uses a streaming approach (no intermediate BAM files), which is faster and uses less disk space.

### Step 5 — Coverage QC (`mosdepth`)

**Tool:** mosdepth

Reports sequencing depth across the exome capture regions. Minimum depth thresholds matter: FACETS (copy number) and DeepSomatic (variant calling) both require adequate coverage to produce reliable results. Low-depth samples typically have higher rates of false positives and false negatives.

### Step 6 — Somatic Variant Calling (`deepsomatic`)

**Tool:** DeepSomatic v1.10.0 (via Singularity)

DeepSomatic uses a convolutional neural network trained on paired tumour-normal pileup images to distinguish true somatic mutations from sequencing noise, germline variants, and alignment artefacts. It processes both samples jointly, which improves sensitivity for low-VAF subclonal mutations.

Output VCF `FILTER` values:
- `PASS` — confident somatic call
- `GERMLINE` — variant also present in normal (germline)
- `RefCall` — insufficient evidence for a variant call

### Step 7 — Variant Filtering (`filter_deepsomatic`)

**Tool:** bcftools

Retains only `PASS` variants meeting minimum VAF (`≥ 0.05` by default) and depth thresholds (`≥ 10×`). These thresholds are configurable in `config.yaml`.

### Step 8 — Allele-Specific Copy Number (`snp_pileup` + `run_facets`)

**Tools:** snp-pileup, FACETS

Cancer genomes are frequently aneuploid — chromosomal regions may be gained (extra copies) or lost (deletions). FACETS estimates, for each chromosomal segment:

- **Total copy number (TCN):** total copies of that region in tumour cells
- **Major copy number (MCN):** copies of the more frequent allele
- **Minor copy number (LCN):** copies of the less frequent allele (can be 0 in loss of heterozygosity)
- **Tumour purity:** fraction of sequenced cells that are tumour

FACETS works by counting allele frequencies at known germline SNP positions (from dbSNP) across both tumour and normal. In normal cells, heterozygous SNPs have a B-allele frequency of ~0.5. In tumour cells with copy number changes, this balance shifts in a predictable way. By fitting a segmentation model to these shifts, FACETS reconstructs the copy number landscape and estimates purity simultaneously.

### Step 9 — Copy Number Validation (`cnaqc`)

**Tool:** CNAqc

CNAqc checks that the FACETS copy number and purity estimates are internally consistent with the somatic mutation VAF distribution. For each copy number state, there is a predicted VAF peak for clonal mutations. CNAqc scores how well the observed VAF peaks match these predictions. A low score suggests the purity or copy number calls may be unreliable.

This is an important quality gate before clonal clustering — if the CN/purity input is wrong, all downstream CCF estimates will be wrong.

### Step 10 — Clonal Clustering and Phylogeny

This is the final stage. Three tools are available, selectable via `clustering_tool` in `config.yaml`.

---

#### Option A: PhylogicNDT (default)

**Rules:** `phylogic_prep` → `phylogic_prepare_sif` → `phylogic_cluster` → `phylogic_build_tree`

PhylogicNDT is a Bayesian framework for clonal reconstruction. It takes somatic mutations annotated with CCF probability distributions (computed from VAF, purity, and local copy number using a binomial likelihood model) and groups them into clusters using a Dirichlet process — a flexible clustering method that automatically infers the number of clusters from the data.

**`phylogic_prep`**
Converts the somatic VCF and FACETS segments into PhylogicNDT's MAF format. For each mutation, a CCF histogram of 101 bins (from 0 to 1) is computed using the binomial likelihood of the observed alt/ref counts given each candidate CCF value, corrected for purity and local copy number. Deeper mutations produce sharper histograms; shallow ones are broader and more uncertain.

**`phylogic_prepare_sif`**
Creates a sample information file (`.sif`) containing the paths to the MAF, segment file, purity value, and timepoint. This is the manifest PhylogicNDT uses to locate all inputs.

**`phylogic_cluster`**
Runs the Dirichlet process clustering. Each cluster in the output represents a subclone — a group of cells that all acquired those mutations at the same point in tumour evolution. The cluster CCF indicates what fraction of tumour cells belong to that subclone. Clusters near CCF = 1.0 are clonal (arose in the founder cell); lower-CCF clusters are subclonal.

**`phylogic_build_tree`**
Given the cluster CCF posteriors, BuildTree infers the most probable phylogenetic relationships between clusters. The output HTML report contains an interactive visualisation of the clone tree.

**Biological interpretation:** In cancers, PhylogicNDT can reveal whether the primary tumour and metastasis share a common ancestor clone, and whether any subclones at diagnosis later dominated at relapse — informative for understanding treatment selection pressure.

# PhylogicNDT edits
Phylogic is part of this repo - that is on purpose. Its old, non-maintained python2 version needed to be edited.
The edits were only made in data/Sample.py and only are about input file format / data reading. no part of the data analysis / algirithm was touched. These edits were done with the help of LLMs, tested and verified working by one human.
the also included .zip files includes the complete, original use base.


---

#### Option B: PyClone6

**Rules:** `pyclone6_prep` → `pyclone6`

PyClone6 is a computationally efficient Bayesian clustering tool using a beta-binomial model, which is well-suited to WES data (as opposed to deep targeted sequencing or WGS). It outperforms many alternatives in the DREAM somatic mutation calling benchmark for WES-specific data.

Unlike PhylogicNDT, PyClone6 performs clustering only — it does not build a phylogenetic tree. Use this if you want fast, robust CCF estimates without tree inference.

---

#### Option C: VIBER (experimental)

**Rules:** `viber_prep` → `viber`

VIBER is a variational inference-based clustering tool. It uses a mean-field variational Bayesian approach (faster than full MCMC but less exact) with a binomial mixture model. The pipeline currently runs VIBER on copy-number neutral (diploid 1+1) regions only, which avoids CCF distortions from aneuploidy at the cost of reducing the number of usable mutations.

---

## Quick Start

### 1. Prepare `samplesheet.csv`

```
sample,type,r1,r2
PATIENT_001,tumor,/data/P001/tumor_R1.fastq.gz,/data/P001/tumor_R2.fastq.gz
PATIENT_001,normal,/data/P001/normal_R1.fastq.gz,/data/P001/normal_R2.fastq.gz
```

Each sample needs exactly one `tumor` row and one `normal` row. The same FASTQ cannot appear twice.

### 2. Edit `config.yaml`

```yaml
output_dir: /path/to/results

reference:
  fasta:     /path/to/hg38.fa
  bwa_ref:   /path/to/hg38.fa
  dbsnp:     /path/to/dbsnp.vcf.gz
  intervals: /path/to/capture.bed   # exome capture BED

clustering_tool: phylogic   # or: pyclone6, viber
```

### 3. Run

```bash
# Dry run (check DAG without executing)
snakemake -n --use-conda --use-singularity --configfile config.yaml

# Local execution
snakemake --cores 16 --use-conda --use-singularity --configfile config.yaml

# SLURM cluster
snakemake --cores 64 --use-conda --use-singularity \
    --executor slurm \
    --default-resources "slurm_account=myproject runtime=480 mem_mb=32000" \
    --configfile config.yaml
```

To rerun a single step after a fix:

```bash
snakemake --cores 4 --use-conda \
    results/PATIENT_001/facets/PATIENT_001_facets_qc.txt --rerun-incomplete
```

---

## Output Structure

```
{output_dir}/
├── {sample}/
│   ├── qc/
│   │   ├── fastqc_trimmed/          FastQC on trimmed reads
│   │   ├── fastp/                   Trimming statistics (JSON + HTML)
│   │   ├── samtools/                Duplicate rate, mapping rate, insert size
│   │   └── mosdepth/                Per-target sequencing depth
│   ├── trimmed/                     Adapter-trimmed FASTQs
│   ├── aligned/
│   │   └── {sample}_{type}_markdup.bam    Analysis-ready BAM + index
│   ├── deepsomatic/
│   │   ├── {sample}_somatic_raw.vcf.gz    All calls (PASS + GERMLINE + RefCall)
│   │   ├── {sample}_somatic_pass.vcf.gz   PASS only, VAF ≥ 0.05, depth ≥ 10
│   │   └── {sample}_somatic_qc.txt        SNV/indel counts, Ti/Tv ratio, VAF summary
│   ├── facets/
│   │   ├── {sample}_facets.rds            Full FACETS result object (R)
│   │   ├── {sample}_cnv_segments.tsv      CN segments (chrom, start, end, TCN, MCN, LCN)
│   │   ├── {sample}_cnv.pdf               Genome-wide CN plot
│   │   └── {sample}_facets_qc.txt         Purity, ploidy, segment count, warnings
│   ├── cnaqc/
│   │   ├── {sample}_cnaqc_qc.txt          Consistency score, PASS/FAIL, purity used
│   │   ├── {sample}_cnaqc_plot.pdf        VAF histogram with expected CN peaks overlaid
│   │   └── {sample}_cnaqc.rds             CNAqc object for downstream analysis
│   ├── phylogic/                    (clustering_tool: phylogic)
│   │   ├── {sample}_phylogic_input.maf    MAF with CCF histograms
│   │   ├── {sample}_phylogic_segments.seg Timing-format CN segments
│   │   ├── {sample}_purity.txt            Purity estimate from FACETS
│   │   ├── {sample}.sif                   Sample information file
│   │   ├── {sample}.cluster_ccfs.txt      Cluster CCF posteriors
│   │   ├── {sample}.mut_ccfs.txt          Per-mutation CCF assignments
│   │   └── {sample}.phylogic_report.html  Interactive clone tree report
│   ├── pyclone6/                    (clustering_tool: pyclone6)
│   │   ├── {sample}_pyclone6_input.tsv    Input: mutations + CN + purity
│   │   ├── {sample}_pyclone6_results.tsv  Per-mutation CCF + cluster assignment
│   │   └── {sample}_pyclone6_plot.pdf     CCF distribution + VAF vs CCF scatter
│   └── viber/                       (clustering_tool: viber)
│       ├── {sample}_viber_input.tsv       Input: CN-neutral mutations
│       ├── {sample}_viber_clusters.tsv    Cluster assignments
│       └── {sample}_*.png                 Cluster visualisation plots
└── multiqc/
    └── multiqc_report.html          Aggregate QC report
```

---

## Interpreting Key Outputs

### `{sample}_facets_qc.txt`
Check `purity` and `ploidy`. A purity below ~0.2 suggests the sample is very heavily contaminated with normal cells — clonal inference will be unreliable. A `WARN` flag indicates FACETS flagged a potential fit issue.

### `{sample}_cnaqc_qc.txt`
A `PASS` means the copy number and purity estimates are consistent with the observed mutation VAF distribution. A `FAIL` warrants manual inspection of the FACETS CN plot and purity estimate before trusting the clustering output.

### `{sample}.phylogic_report.html`
Open in a browser. Contains:
- A scatter plot of mutations by CCF, coloured by cluster
- A phylogenetic tree of the clusters
- Arm-level copy number events placed on the tree
- Driver gene annotations where applicable

Clusters near CCF = 1.0 are **clonal** (present in all tumour cells, arose early). Clusters with lower CCF are **subclonal** (arose later in a subset of cells). The tree shows which subclones descended from which.

---

## Common Failure Modes

**FACETS returns `purity = NA`**
Too few heterozygous SNPs per segment. Causes: wrong capture BED, low depth, or chromosome naming mismatch (`chr1` vs `1`) between the BED, BAM, and dbSNP VCF. Check `nhet` column in `*_cnv_segments.tsv`.

**PhylogicNDT `KeyError: 'Start'` or similar**
The segment file format must use `.seg` extension (not `.tsv`) — PhylogicNDT's format auto-detection is purely extension-based. The pipeline handles this correctly; this error should not appear unless the Snakemake rule is modified.

**PhylogicNDT `KeyError: '23'`**
FACETS encodes chromosome X as `23`. The build script remaps this to `X` for PhylogicNDT compatibility.

**CNAqc fails with row-mismatch error**
Usually a downstream consequence of FACETS `NA` purity — fix the pileup/BED issue first.

**DeepSomatic Singularity bind mount error**
The reference directory must be explicitly bound with `-B`. Multiple paths require separate `-B` flags.

**`Fewer than 10 mutations after filtering`**
The depth filter (`dp_v >= 25`) or the copy-number neutral filter in VIBER removed too many mutations. Check the somatic QC output and mosdepth coverage summary.

---

## Reference File Requirements

| File | Tool | Preparation |
|------|------|-------------|
| `reference.fasta` | BWA-MEM2, DeepSomatic, samtools | `samtools faidx genome.fa` + `bwa-mem2 index genome.fa` |
| `reference.dbsnp` | snp-pileup | `bgzip dbsnp.vcf && tabix -p vcf dbsnp.vcf.gz` |
| `reference.intervals` | DeepSomatic, mosdepth, snp-pileup | Exome capture BED matching your library prep kit |

Chromosome naming must be consistent across all reference files (`chr1`-style for hg38).

---

## Clustering Tool Comparison

| Tool | Method | Tree inference | Best for |
|------|--------|---------------|---------|
| PhylogicNDT | Dirichlet process (MCMC) | Yes | Full clonal reconstruction including phylogeny |
| PyClone6 | Beta-binomial (variational) | No | Fast, robust CCF estimation; top WES benchmark performer |
| VIBER | Binomial mixture (variational) | No | Experimental; CN-neutral mutations only |

PhylogicNDT is the default and recommended tool. PyClone6 is the best alternative if runtime is a concern or tree inference is not needed. VIBER is experimental.

---

## Software Environment Notes

PhylogicNDT requires **Python 2.7** and `scikit-learn=0.18.1` — it predates Python 3. The dedicated `envs/phylogic.yaml` conda environment handles this. Do not attempt to run PhylogicNDT in any other environment.

DeepSomatic has no conda package and runs via Singularity. The container image is specified in `config.yaml` under `singularity.deepsomatic`.

All other tools are managed via conda environments defined in `envs/`.

---

## Tool Citations

| Tool | Citation |
|------|---------|
| fastp | Chen et al., *Bioinformatics* 34:i884 (2018) |
| FastQC / MultiQC | Andrews (2010); Ewels et al., *Bioinformatics* 32:3047 (2016) |
| BWA-MEM2 | Vasimuddin et al., *IPDPS* pp.314–324 (2019) |
| samtools | Danecek et al., *GigaScience* 10:giab008 (2021) |
| mosdepth | Pedersen & Quinlan, *Bioinformatics* 34:867 (2018) |
| DeepSomatic | Park, Cook, Chang et al., *Nat Biotechnol* (2025) |
| FACETS | Shen & Seshan, *Nucleic Acids Res* 44:e131 (2016) |
| CNAqc | Antonello, Bergamin et al., *Genome Biol* 25:38 (2024) |
| PhylogicNDT | Garofalo et al., *Nat Cancer* 4:731 (2023) |
| PyClone6 | Gillis & Roth, *BMC Bioinformatics* 21:571 (2020) |
| VIBER | Caravagna et al.,not published yet (2026) (caravagnalab) |
