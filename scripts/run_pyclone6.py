"""
scripts/run_pyclone6.py — Snakemake script: directive
Converts FACETS RDS + somatic PASS VCF → PyClone6 input TSV, then runs PyClone6.
"""
import os, subprocess
import pandas as pd

log_path    = snakemake.log[0]
vcf_file    = snakemake.input["vcf"]
seg_file    = snakemake.input["seg"]
rds_file    = snakemake.input["rds"]
out_results = snakemake.output["results"]
out_plot    = snakemake.output["plot"]
n_clusters  = snakemake.params["n_clusters"]
n_restarts  = snakemake.params["n_restarts"]
num_iters   = snakemake.params["num_iters"]
density     = snakemake.params["density"]
outdir      = snakemake.params["outdir"]
sample_name = snakemake.params["sample_name"]

os.makedirs(outdir, exist_ok=True)

def run(cmd):
    with open(log_path, "a") as lf:
        lf.write(f"\n$ {cmd}\n")
        r = subprocess.run(cmd, shell=True, text=True,
                           stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")

with open(log_path, "w") as lf:
    lf.write(f"=== PyClone6: {sample_name} ===\n")

# ── 1. Extract purity from FACETS RDS ────────────────────────────────────────
purity_file = os.path.join(outdir, f"{sample_name}_purity.txt")
run(f"""Rscript -e "obj <- readRDS('{rds_file}'); cat(obj\\$fit\\$purity, '\\n')" > {purity_file}""")
with open(purity_file) as f:
    purity = float(f.read().strip())

with open(log_path, "a") as lf:
    lf.write(f"Purity from FACETS: {purity:.4f}\n")

# ── 2. Parse somatic VCF → ref/alt counts ────────────────────────────────────
result = subprocess.run(
    f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t[%AD]\\t[%DP]\\n' {vcf_file}",
    shell=True, capture_output=True, text=True
)

seg = pd.read_csv(seg_file, sep="\t")
seg["chrom_str"] = "chr" + seg["chrom"].astype(str).str.replace(r"^chr", "", regex=True)

def get_cn(chrom, pos):
    m = seg[(seg["chrom_str"] == chrom) &
            (seg["loc.start"] <= pos) &
            (seg["loc.end"]   >= pos)]
    if m.empty:
        return 2, 1   # diploid default
    row = m.iloc[0]
    major = max(int(row.get("mcn", 1)), 0)
    minor = max(int(row.get("lcn", 0)), 0)
    # Ensure major >= minor
    return max(major, minor), min(major, minor)

records = []
for line in result.stdout.strip().split("\n"):
    if not line:
        continue
    chrom, pos, ref, alt, ad, dp = line.split("\t")
    pos = int(pos)
    dp  = int(dp)
    if dp < 10:
        continue
    parts = ad.split(",")
    ref_c = int(parts[0])
    alt_c = int(parts[1]) if len(parts) > 1 else 0
    major, minor = get_cn(chrom, pos)
    records.append({
        "mutation_id":    f"{chrom}:{pos}:{ref}>{alt}",
        "sample_id":      sample_name,
        "ref_counts":     ref_c,
        "alt_counts":     alt_c,
        "normal_cn":      2,
        "minor_cn":       minor,
        "major_cn":       major,
        "tumour_content": round(purity, 4),
    })

tsv_path = os.path.join(outdir, f"{sample_name}_pyclone6_input.tsv")
df = pd.DataFrame(records)
df.to_csv(tsv_path, sep="\t", index=False)

with open(log_path, "a") as lf:
    lf.write(f"Mutations in PyClone6 input: {len(df)}\n")

if len(df) < 10:
    raise RuntimeError(f"Only {len(df)} mutations after filtering — too few for PyClone6")

# ── 3. Run PyClone6 ───────────────────────────────────────────────────────────
run(f"""pyclone6 run_analysis \
    --in_files {tsv_path} \
    --out_dir {outdir} \
    --num_clusters {n_clusters} \
    --num_restarts {n_restarts} \
    --num_iters {num_iters} \
    --density {density}""")

run(f"pyclone6 build_results_file --in_dir {outdir} --out_file {out_results}")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
run(f"Rscript scripts/plot_pyclone6.R {out_results} {out_plot} {sample_name}")
