"""
Script to generate ultra-high-resolution energy level comparison plots (600 DPI)
for Thesis Sections 4.1, 4.2, and 4.3 using the new optimization results.

The script calls YZS_lib.create_energy_level_plot (the plotting function from YZS_fit)
and saves the images into their corresponding test folders:
1. Thesis 4.1: DyCl3 -> test/(Crosswhite H M et al, 1961)/dy.png
2. Thesis 4.2: Er3+:YVO4 -> test/(Capobianco J A et al, 1997)/er4.png
3. Thesis 4.3: Eu3+:YVO4 -> test/(Brecher C et al, 1967)/eu.png
"""

import os
import sys
import re
from unittest.mock import MagicMock

# Safely mock PyCrystalField to avoid dependency issues when importing YZS_lib
sys.modules['PyCrystalField'] = MagicMock()

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import YZS_lib


def parse_result_file(file_path):
    """
    Parse THNRG and EXNRG dictionaries from a YZS Result.txt file.
    """
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    start_idx = None
    for i, line in enumerate(lines):
        if 'Comparison between theoretical and experimental energies:' in line:
            start_idx = i + 2  # skip table header line
            break
            
    if start_idx is None:
        raise ValueError(f"Could not find energy comparison section in {file_path}")
        
    th_nrg = {}
    ex_nrg = {}
    current_term = None
    
    for line in lines[start_idx:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith('***') or s.startswith('==='):
            break
            
        # Term identifier header (e.g., '6H15/2', '4I15/2', '7F1')
        if ':' not in s and not re.search(r'^-?\d+\.\d+', s):
            current_term = s
            th_nrg[current_term] = []
            ex_nrg[current_term] = []
            continue
            
        if current_term and ':' in s:
            parts = s.split(':')
            th_val = float(parts[0].strip())
            ex_str = parts[1].strip()
            ex_val = None if ex_str == 'None' else float(ex_str)
            th_nrg[current_term].append(th_val)
            ex_nrg[current_term].append(ex_val)
            
    return th_nrg, ex_nrg


def plot_and_save(result_file, output_image_path, title_prefix=""):
    """
    Plot energy levels using YZS_lib.create_energy_level_plot and save at 600 DPI.
    """
    print(f"[{title_prefix}] Parsing: {result_file}")
    th_nrg, ex_nrg = parse_result_file(result_file)
    print(f"[{title_prefix}] Found {len(th_nrg)} spectral terms: {list(th_nrg.keys())}")
    
    # Intercept plt.show to save the figure with high resolution
    def save_and_close():
        fig = plt.gcf()
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        fig.savefig(output_image_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        
    plt.show = save_and_close
    
    # Call YZS plotting function
    YZS_lib.create_energy_level_plot(th_nrg, ex_nrg)
    
    file_size_kb = os.path.getsize(output_image_path) / 1024
    print(f"[{title_prefix}] Saved high-res image (600 DPI): {output_image_path} ({file_size_kb:.1f} KB)")


def main():
    base_dir = os.path.dirname(__file__)
    
    tasks = [
        {
            "name": "Thesis 4.1: DyCl3",
            "result": os.path.join(base_dir, "(Crosswhite H M et al, 1961)", "Result.txt"),
            "output": os.path.join(base_dir, "(Crosswhite H M et al, 1961)", "dy.png")
        },
        {
            "name": "Thesis 4.2: Er3+:YVO4",
            "result": os.path.join(base_dir, "(Capobianco J A et al, 1997)", "Result2.txt"),
            "output": os.path.join(base_dir, "(Capobianco J A et al, 1997)", "er4.png")
        },
        {
            "name": "Thesis 4.3: Eu3+:YVO4",
            "result": os.path.join(base_dir, "(Brecher C et al, 1967)", "Result.txt"),
            "output": os.path.join(base_dir, "(Brecher C et al, 1967)", "eu.png")
        }
    ]
    
    print("=" * 70)
    print("Starting generation of high-resolution energy level comparison plots...")
    print("=" * 70)
    
    for task in tasks:
        plot_and_save(task["result"], task["output"], title_prefix=task["name"])
        print("-" * 70)
        
    print("All 3 high-resolution plots successfully created!")


if __name__ == "__main__":
    main()
