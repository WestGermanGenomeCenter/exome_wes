#!/usr/bin/env bash
# setup/offline_prefetch.sh
# Run ONCE on a machine with internet access. Pulls Singularity images and
# conda-packs all environments for transfer to an offline HPC.
#
# Usage:
#   bash setup/offline_prefetch.sh --outdir /data/pipeline_cache
#
# On the offline HPC:
#   snakemake --cores 32 --use-singularity --use-conda \
#       --singularity-prefix /data/pipeline_cache/sif \
#       --conda-prefix /data/pipeline_cache/conda_envs \
#       --configfile config.yaml

set -euo pipefail

OUTDIR="/data/pipeline_cache"
while [[ $# -gt 0 ]]; do
    case $1 in --outdir) OUTDIR="$2"; shift 2 ;; *) echo "Unknown: $1"; exit 1 ;; esac
done

SIF_DIR="${OUTDIR}/sif"
ENV_DIR="${OUTDIR}/conda_envs"
PACK_DIR="${OUTDIR}/conda_packs"
mkdir -p "${SIF_DIR}" "${ENV_DIR}" "${PACK_DIR}"
echo "=== Offline prefetch → ${OUTDIR} ==="

# ── Singularity images ────────────────────────────────────────────────────────
echo "--- Pulling Singularity images ---"
[ -f "${SIF_DIR}/deepsomatic_1.10.0.sif" ] \
    && echo "  [skip] deepsomatic already exists" \
    || singularity pull "${SIF_DIR}/deepsomatic_1.10.0.sif" \
         docker://google/deepsomatic:1.10.0

echo ""
echo "Set in config.yaml:"
echo "  singularity:"
echo "    deepsomatic: ${SIF_DIR}/deepsomatic_1.10.0.sif"

# ── Conda environments ────────────────────────────────────────────────────────
echo ""
echo "--- Building and packing conda environments ---"
conda install -y -c conda-forge conda-pack 2>/dev/null || true

ENVS=(
    envs/qc.yaml
    envs/align.yaml
    envs/gatk.yaml
    envs/bcftools.yaml
    envs/facets.yaml
    envs/cnaqc.yaml
    envs/phylogic.yaml
    envs/pyclone6.yaml
)

for yaml in "${ENVS[@]}"; do
    env_name=$(grep "^name:" "${yaml}" | awk '{print $2}')
    pack_file="${PACK_DIR}/${env_name}.tar.gz"
    env_path="${ENV_DIR}/${env_name}"

    if [[ -f "${pack_file}" ]]; then
        echo "  [skip] ${env_name} already packed"
        continue
    fi
    echo "  Building: ${env_name}"
    conda env create --prefix "${env_path}" -f "${yaml}" -q
    echo "  Packing:  ${env_name} → ${pack_file}"
    conda pack --prefix "${env_path}" -o "${pack_file}" -q
    echo "  Done: ${env_name}"
done

echo ""
echo "=== Prefetch complete. Transfer ${OUTDIR} to HPC, then: ==="
echo ""
echo "  # Unpack envs on HPC (run once per node type):"
echo "  for f in ${PACK_DIR}/*.tar.gz; do"
echo "    name=\$(basename \"\$f\" .tar.gz)"
echo "    mkdir -p ${ENV_DIR}/\$name"
echo "    tar -xzf \"\$f\" -C ${ENV_DIR}/\$name"
echo "    conda-unpack --prefix ${ENV_DIR}/\$name"
echo "  done"
echo ""
echo "  # Run pipeline:"
echo "  snakemake --cores 32 --use-singularity --use-conda \\"
echo "      --singularity-prefix ${SIF_DIR} \\"
echo "      --conda-prefix ${ENV_DIR} \\"
echo "      --configfile config.yaml"
