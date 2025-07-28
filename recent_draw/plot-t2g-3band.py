#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_all_bands_in_one_figure(filename):
    """
    Plots all bands from a given data file onto a single figure.
    Column 0 is the x-axis. Even-indexed columns (2, 4, 6...) are plotted as y-axis bands.
    This corresponds to gnuplot's `u 1:3, "" u 1:5, ...` syntax.
    """
    try:
        data = np.loadtxt(filename)
    except Exception as e:
        print(f"Cannot read file {filename}: {e}")
        return

    num_cols = data.shape[1]
    if num_cols < 3:
        print(f"Warning: {filename} has fewer than 3 columns, cannot plot.")
        return

    plt.figure(figsize=(12, 7))
    
    x_axis = data[:, 0]
    
    # Iterate through even-indexed columns for y-values, starting from index 2
    for i in range(2, num_cols, 2):
        plt.plot(x_axis, data[:, i], label=f'Column {i + 1}')

    plt.title(f'All Bands - {os.path.basename(filename)}')
    plt.xlabel('Frequency (X-axis, Column 1)')
    plt.ylabel('Real Part (Y-axis)')
    plt.legend()
    plt.xlim(0, 15)
    plt.ylim(-1.5, 0)
    plt.grid(True)
    
    output_filename = f"all_bands_real_{os.path.basename(filename)}.png"
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Chart saved to: {output_filename}")

def main():
    """
    Finds all '*.inp.*' files in the current directory and plots them.
    """
    # Use glob to find all files matching the pattern *.inp.*
    files = [f for f in glob.glob("*.inp.*") if not f.endswith('.png')]
    
    if not files:
        print("No matching files '*.inp.*' found in the current directory.")
        return
    
    print(f"Found {len(files)} files to process...")
    
    # Process each file
    for filename in sorted(files):
        print(f"Processing file: {filename}")
        plot_all_bands_in_one_figure(filename)
    
    print("\nAll files processing completed!")

if __name__ == "__main__":
    main()

