#!/usr/bin/env python3
"""
Add variant allele frequency (VAF) to pyclone6 results file.

Usage:
    python3 scripts/add_vaf_to_pyclone6_results.py \
        <input_tsv> <results_tsv> <output_tsv>

Args:
    input_tsv:   Original pyclone6 input file (contains ref_counts, alt_counts)
    results_tsv: PyClone6 results file (from pyclone-vi write-results-file)
    output_tsv:  Output results file with VAF column added
"""

import sys
import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def validate_input_file(df, filepath):
    """Check that input file has required columns."""
    required_cols = {'mutation_id', 'ref_counts', 'alt_counts'}
    missing = required_cols - set(df.columns)
    
    if missing:
        logger.error(
            f"Input file '{filepath}' missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
        return False
    
    # Check for non-negative counts
    if (df['ref_counts'] < 0).any() or (df['alt_counts'] < 0).any():
        logger.error("Input file contains negative ref_counts or alt_counts")
        return False
    
    # Check for rows with zero total counts
    total_counts = df['ref_counts'] + df['alt_counts']
    zero_count_rows = (total_counts == 0).sum()
    if zero_count_rows > 0:
        logger.warning(
            f"Found {zero_count_rows} mutations with zero total counts "
            "(ref_counts + alt_counts = 0). VAF will be NaN for these."
        )
    
    return True


def validate_results_file(df, filepath):
    """Check that results file has required columns."""
    required_cols = {'mutation_id', 'cluster_id', 'cellular_prevalence'}
    missing = required_cols - set(df.columns)
    
    if missing:
        logger.error(
            f"Results file '{filepath}' missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
        return False
    
    return True


def calculate_vaf(input_df):
    """Calculate VAF from ref and alt counts."""
    total_counts = input_df['ref_counts'] + input_df['alt_counts']
    vaf = input_df['alt_counts'] / total_counts
    
    return vaf


def merge_vaf_to_results(input_df, results_df):
    """Merge VAF into results dataframe."""
    # Calculate VAF
    logger.info("Calculating VAF from ref_counts and alt_counts...")
    input_df['variant_allele_frequency'] = calculate_vaf(input_df)
    
    # Prepare input data for merge (keep only necessary columns)
    input_subset = input_df[['mutation_id', 'variant_allele_frequency']].copy()
    
    # Merge on mutation_id
    logger.info("Merging VAF with results file...")
    merged_df = results_df.merge(
        input_subset,
        on='mutation_id',
        how='left'
    )
    
    # Check for unmatched mutations
    unmatched_results = merged_df['variant_allele_frequency'].isna().sum()
    if unmatched_results > 0:
        logger.warning(
            f"Found {unmatched_results} mutations in results file that were not in input file. "
            "VAF will be NaN for these rows."
        )
    
    unmatched_input = len(input_df) - len(results_df.merge(
        input_subset,
        on='mutation_id',
        how='inner'
    ))
    if unmatched_input > 0:
        logger.warning(
            f"Found {unmatched_input} mutations in input file that were not in results file. "
            "These will not appear in output."
        )
    
    return merged_df


def main():
    """Main execution function."""
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    
    input_tsv = sys.argv[1]
    results_tsv = sys.argv[2]
    output_tsv = sys.argv[3]
    
    # Validate file paths
    for filepath in [input_tsv, results_tsv]:
        if not Path(filepath).exists():
            logger.error(f"File not found: {filepath}")
            sys.exit(1)
    
    try:
        # Read files
        logger.info(f"Reading input file: {input_tsv}")
        input_df = pd.read_csv(input_tsv, sep='\t')
        
        logger.info(f"Reading results file: {results_tsv}")
        results_df = pd.read_csv(results_tsv, sep='\t')
        
        # Validate input files
        logger.info("Validating input file...")
        if not validate_input_file(input_df, input_tsv):
            sys.exit(1)
        
        logger.info("Validating results file...")
        if not validate_results_file(results_df, results_tsv):
            sys.exit(1)
        
        logger.info(f"Input file: {len(input_df)} mutations")
        logger.info(f"Results file: {len(results_df)} mutations")
        
        # Merge VAF into results
        merged_df = merge_vaf_to_results(input_df, results_df)
        
        # Write output
        logger.info(f"Writing output file: {output_tsv}")
        merged_df.to_csv(output_tsv, sep='\t', index=False)
        
        # Summary statistics
        logger.info(f"Output file: {len(merged_df)} mutations with {len(merged_df.columns)} columns")
        
        # Check VAF values
        vaf_stats = merged_df['variant_allele_frequency'].describe()
        logger.info(f"VAF statistics:\n{vaf_stats}")
        
        logger.info("Successfully completed!")
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
