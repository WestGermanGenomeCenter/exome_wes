#!/usr/bin/env python3
"""
Plot Orchard phylogenetic tree from NPZ output.
Usage: python3 plot_orchard_tree.py <npz_file> <output_pdf>
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from Bio import Phylo
from io import StringIO
import json

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 plot_orchard_tree.py <npz_file> <output_pdf>")
        sys.exit(1)
    
    npz_file = sys.argv[1]
    output_pdf = sys.argv[2]
    
    print(f"[INFO] Loading Orchard NPZ: {npz_file}")
    
    try:
        data = np.load(npz_file, allow_pickle=True)
    except Exception as e:
        print(f"[ERROR] Failed to load NPZ: {e}")
        sys.exit(1)
    
    # Print available keys
    keys = list(data.keys())
    print(f"[INFO] NPZ keys found: {', '.join(keys)}")
    
    # Check for newick key
    if 'newick' not in keys:
        print("[ERROR] 'newick' key not found in NPZ file")
        sys.exit(1)
    
    # Extract newick strings
    newick_array = data['newick']
    print(f"[INFO] Newick array shape: {newick_array.shape}, dtype: {newick_array.dtype}")
    
    # Get first (best) tree
    best_tree_newick = str(newick_array[0])
    print(f"[INFO] Selected tree: {best_tree_newick}")
    
    # Parse tree using BioPython
    try:
        tree = Phylo.read(StringIO(best_tree_newick), "newick")
    except Exception as e:
        print(f"[ERROR] Failed to parse Newick string: {e}")
        sys.exit(1)
    
    # Create figure and plot
    print(f"[INFO] Generating phylogenetic tree plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        show_confidence=False
    )
    
    ax.set_title("Orchard Phylogenetic Tree", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    print(f"[INFO] Saving PDF: {output_pdf}")
    try:
        plt.savefig(output_pdf, dpi=300, format='pdf')
        print(f"[INFO] PDF successfully saved to {output_pdf}")
    except Exception as e:
        print(f"[ERROR] Failed to save PDF: {e}")
        sys.exit(1)
    
    plt.close()
    print("[INFO] Process completed successfully.")

if __name__ == "__main__":
    main()
