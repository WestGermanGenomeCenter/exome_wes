"""
scripts/run_phylogic_cluster.py — Snakemake script: directive
Converts FACETS output + somatic VCF → PhylogicNDT MAF, runs PhylogicNDT Cluster.
PhylogicNDT is not a Python package — it is called as a script from the
cloned repo directory set in config.yaml under phylogic.phylogicndt_dir.
Run setup/install_phylogic.sh once to clone and install dependencies.
"""
import os, subprocess
import pandas as pd

log_path         = snakemake.log[0]
vcf_file         = snakemake.input["vcf"]
seg_file         = snakemake.input["seg"]
rds_file         = snakemake.input["rds"]
out_post         = snakemake.output["posteriors"]
out_maf          = snakemake.output["maf"]
n_iter           = snakemake.params["n_iter"]
phylogicndt_dir  = snakemake.params["phylogicndt_dir"]
sample_name      = snakemake.params["sample_name"]
outdir           = snakemake.params["outdir"]

os.makedirs(outdir, exist_ok=True)

phylogic_py = os.path.join(phylogicndt_dir, "PhylogicNDT.py")
if not os.path.exists(phylogic_py):
    raise FileNotFoundError(
        f"PhylogicNDT.py not found at {phylogic_py}.\n"
        f"Run: bash setup/install_phylogic.sh --dest tools\n"
        f"Then set phylogic.phylogicndt_dir in config.yaml"
    )

def run(cmd):
    with open(log_path, "a") as lf:
        lf.write(f"\n$ {cmd}\n")
        r = subprocess.run(cmd, shell=True, text=True,
                           stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")

with open(log_path, "w") as lf:
    lf.write(f"=== PhylogicNDT Cluster: {sample_name} ===\n")
    lf.write(f"PhylogicNDT.py: {phylogic_py}\n")

# ── 1. Extract purity from FACETS RDS ────────────────────────────────────────
purity_file = os.path.join(outdir, f"{sample_name}_purity.txt")
run(f"""Rscript -e "obj <- readRDS('{rds_file}'); cat(obj\\$fit\\$purity, '\\n')" > {purity_file}""")
with open(purity_file) as f:
    purity = float(f.read().strip())

with open(log_path, "a") as lf:
    lf.write(f"Purity from FACETS: {purity:.4f}\n")

# ── 2. Parse VCF + attach CN → PhylogicNDT MAF format ────────────────────────
# PhylogicNDT MAF required cols (when using --maf_input_type calc_ccf):
#   Hugo_Symbol, Chromosome, Start_position, Reference_Allele,
#   Tumor_Seq_Allele2, ref_count, alt_count, local_cn_a1, local_cn_a2
result = subprocess.run(
    f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t[%AD]\\t[%DP]\\n' {vcf_file}",
    shell=True, capture_output=True, text=True
)

seg = pd.read_csv(seg_file, sep="\t")
seg["chrom_str"] = "chr" + seg["chrom"].astype(str).str.replace(r"^chr", "", regex=True)

def get_cn(chrom, pos):
    """Return (major_cn, minor_cn); default diploid (1,1) if no overlap."""
    m = seg[(seg["chrom_str"] == chrom) &
            (seg["loc.start"] <= pos) &
            (seg["loc.end"]   >= pos)]
    if m.empty:
        return 1, 1
    row = m.iloc[0]
    major = max(int(row.get("mcn", 1)), 0)
    minor = max(int(row.get("lcn", 0)), 0)
    return max(major, minor), min(major, minor)

rows = []
for line in result.stdout.strip().split("\n"):
    if not line:
        continue
    fields = line.split("\t")
    if len(fields) < 6:
        continue
    chrom, pos, ref, alt, ad, dp = fields[:6]
    pos = int(pos)
    if int(dp) < 10:
        continue
    parts = ad.split(",")
    ref_c = int(parts[0])
    alt_c = int(parts[1]) if len(parts) > 1 else 0
    cn_a1, cn_a2 = get_cn(chrom, pos)
    rows.append({
        "Hugo_Symbol":       "Unknown",
        "Chromosome":        chrom.replace("chr", ""),
        "Start_position":    pos,
        "Reference_Allele":  ref,
        "Tumor_Seq_Allele2": alt,
        "ref_count":         ref_c,
        "alt_count":         alt_c,
        "local_cn_a1":       cn_a1,
        "local_cn_a2":       cn_a2,
    })

maf_df = pd.DataFrame(rows)
maf_df.to_csv(out_maf, sep="\t", index=False)

with open(log_path, "a") as lf:
    lf.write(f"MAF rows after depth filter: {len(maf_df)}\n")

if len(maf_df) < 10:
    raise RuntimeError(
        f"Only {len(maf_df)} mutations after depth filtering — "
        "too few for PhylogicNDT (need >= 10)"
    )

# ── 3. Write PhylogicNDT .sif manifest ───────────────────────────────────────
# PhylogicNDT uses a TSV manifest (.sif) to describe each sample
sif_path = os.path.join(outdir, f"{sample_name}.sif")
with open(sif_path, "w") as f:
    f.write("sample_id\tmaf_fn\tseg_fn\tpurity\n")
    f.write(f"{sample_name}\t{os.path.abspath(out_maf)}\t"
            f"{os.path.abspath(seg_file)}\t{purity:.4f}\n")

# ── 4. Run PhylogicNDT Cluster ────────────────────────────────────────────────
run(f"""python {phylogic_py} Cluster \
    -i {sample_name} \
    -sif {sif_path} \
    --maf_input_type calc_ccf \
    -n {n_iter} \
    --seed 42""")

# ── 5. Move outputs to expected paths ─────────────────────────────────────────
# PhylogicNDT writes to CWD using sample_name as prefix
moved = False
for suffix in ["cluster_posteriors.tsv", "mut_ccfs.txt"]:
    candidate = f"{sample_name}.{suffix}"
    if os.path.exists(candidate):
        run(f"mv {candidate} {out_post}")
        moved = True
        break

if not moved:
    raise RuntimeError(
        f"PhylogicNDT Cluster output not found in CWD.\n"
        f"Expected: {sample_name}.cluster_posteriors.tsv\n"
        f"Check log: {log_path}"
    )

with open(log_path, "a") as lf:
    lf.write("PhylogicNDT Cluster complete.\n")
