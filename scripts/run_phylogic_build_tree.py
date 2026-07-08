"""
scripts/run_phylogic_build_tree.py — Snakemake script: directive
Runs PhylogicNDT BuildTree from Cluster posteriors.
"""
import os, subprocess

log_path        = snakemake.log[0]
posteriors      = snakemake.input["posteriors"]
maf_file        = snakemake.input["maf"]
seg_file        = snakemake.input["seg"]
out_trees       = snakemake.output["trees"]
out_html        = snakemake.output["html"]
phylogicndt_dir = snakemake.params["phylogicndt_dir"]
sample_name     = snakemake.params["sample_name"]
outdir          = snakemake.params["outdir"]

phylogic_py = os.path.join(phylogicndt_dir, "PhylogicNDT.py")
if not os.path.exists(phylogic_py):
    raise FileNotFoundError(
        f"PhylogicNDT.py not found at {phylogic_py}.\n"
        f"Run: bash setup/install_phylogic.sh --dest tools"
    )

sif_path = os.path.join(outdir, f"{sample_name}.sif")

def run(cmd):
    with open(log_path, "a") as lf:
        lf.write(f"\n$ {cmd}\n")
        r = subprocess.run(cmd, shell=True, text=True,
                           stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"Command failed (exit {r.returncode}): {cmd}")

with open(log_path, "w") as lf:
    lf.write(f"=== PhylogicNDT BuildTree: {sample_name} ===\n")
    lf.write(f"PhylogicNDT.py: {phylogic_py}\n")

run(f"""python {phylogic_py} BuildTree \
    -i {sample_name} \
    -sif {sif_path} \
    --seed 42""")

# Move tree posteriors
moved = False
for suffix in ["tree_posteriors.tsv", "mutation_tree_posteriors.tsv"]:
    candidate = f"{sample_name}.{suffix}"
    if os.path.exists(candidate):
        run(f"mv {candidate} {out_trees}")
        moved = True
        break

if not moved:
    raise RuntimeError(
        f"BuildTree output not found in CWD.\n"
        f"Expected: {sample_name}.tree_posteriors.tsv\n"
        f"Check log: {log_path}"
    )

# HTML visualisation (may or may not be produced depending on PhylogicNDT version)
html_candidate = f"{sample_name}_html_tree.html"
if os.path.exists(html_candidate):
    run(f"mv {html_candidate} {out_html}")
else:
    with open(out_html, "w") as f:
        f.write(
            f"<html><body><pre>PhylogicNDT BuildTree complete: {sample_name}.\n"
            f"Tree posteriors: {os.path.abspath(out_trees)}</pre></body></html>\n"
        )

with open(log_path, "a") as lf:
    lf.write("PhylogicNDT BuildTree complete.\n")
