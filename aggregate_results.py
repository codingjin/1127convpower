#!/usr/bin/env python3
"""
Aggregate performance and energy results from kernel output files.

Scans kernel_outputs/case{M}/powercap{N}/ directories and extracts:
- GFLOP/s from [Per-Iteration Performance] section
- Mean energy per iteration (mJ) from [Per-Iteration Energy] section

Creates results.csv in each powercap directory with format:
id,perf(GFLOP/s),energy(mJ)

Note: Kernel numbering starts from 1 and restarts within each case.
"""

import os
import re
import csv
from pathlib import Path


def parse_output_file(file_path):
    """
    Parse a single output_kernel{K}.txt file and extract GFLOP/s and energy.

    Returns:
        tuple: (gflops, energy_mj) or (None, None) if parsing fails
    """
    gflops = None
    energy_mj = None

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        # Extract GFLOP/s from [Per-Iteration Performance] section
        # Pattern: "  GFLOP/s: 123.456"
        gflops_match = re.search(r'GFLOP/s:\s+([\d.]+)', content)
        if gflops_match:
            gflops = float(gflops_match.group(1))

        # Extract energy from [Per-Iteration Energy] section
        # Pattern: "  Mean energy per iteration: 12.345678000 mJ"
        energy_match = re.search(r'Mean energy per iteration:\s+([\d.]+)\s+mJ', content)
        if energy_match:
            energy_mj = float(energy_match.group(1))

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None

    return gflops, energy_mj


def aggregate_powercap_directory(powercap_dir):
    """
    Aggregate all kernel results in a powercap directory.

    Args:
        powercap_dir: Path to kernel_outputs/case{M}/powercap{N}/

    Returns:
        list: List of (kernel_id, gflops, energy_mj) tuples, sorted by kernel_id
    """
    results = []

    # Find all output_kernel{K}.txt files (K starts from 1)
    output_files = sorted(powercap_dir.glob('output_kernel*.txt'))

    for output_file in output_files:
        # Extract kernel number from filename: output_kernel1.txt -> 1
        match = re.search(r'output_kernel(\d+)\.txt', output_file.name)
        if not match:
            continue

        kernel_num = int(match.group(1))
        kernel_id = kernel_num  # id = K (starts from 1)

        # Parse the file
        gflops, energy_mj = parse_output_file(output_file)

        if gflops is not None and energy_mj is not None:
            results.append((kernel_id, gflops, energy_mj))
        else:
            print(f"Warning: Could not parse {output_file}")

    # Sort by kernel_id
    results.sort(key=lambda x: x[0])

    return results


def write_csv(powercap_dir, results):
    """
    Write results to results.csv in the powercap directory.

    Args:
        powercap_dir: Path to kernel_outputs/case{M}/powercap{N}/
        results: List of (kernel_id, gflops, energy_mj) tuples
    """
    csv_path = powercap_dir / 'results.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['id', 'perf(GFLOP/s)', 'energy(mJ)'])

        # Write data
        for kernel_id, gflops, energy_mj in results:
            # Round GFLOP/s to integer
            perf_int = round(gflops)
            # Format energy to 2 decimal places
            energy_formatted = round(energy_mj, 2)

            writer.writerow([kernel_id, perf_int, energy_formatted])

    print(f"✓ Created {csv_path} ({len(results)} kernels)")


def main():
    """Main function to aggregate all results."""
    import sys

    # Check if specific case_id provided as command-line argument
    target_case_id = sys.argv[1] if len(sys.argv) > 1 else None

    kernel_outputs_dir = Path('kernel_outputs')

    if not kernel_outputs_dir.exists():
        print(f"Error: {kernel_outputs_dir} directory not found")
        print("Please run from the project root directory")
        return

    # Find all case directories (or specific case if provided)
    if target_case_id:
        case_dirs = [kernel_outputs_dir / target_case_id]
        if not case_dirs[0].exists():
            print(f"Error: {case_dirs[0]} directory not found")
            return
    else:
        case_dirs = sorted(kernel_outputs_dir.glob('case*'))

    if not case_dirs:
        print(f"No case directories found in {kernel_outputs_dir}")
        return

    total_csvs = 0
    total_kernels = 0

    if target_case_id:
        print(f"Aggregating results for {target_case_id}...\n")
    else:
        print("Aggregating results for all cases...\n")

    # Process each case directory
    for case_dir in case_dirs:
        case_name = case_dir.name
        print(f"Processing {case_name}/")

        # Find all powercap directories
        powercap_dirs = sorted(case_dir.glob('powercap*'))

        for powercap_dir in powercap_dirs:
            powercap_name = powercap_dir.name

            # Aggregate results for this powercap
            results = aggregate_powercap_directory(powercap_dir)

            if results:
                # Write CSV
                write_csv(powercap_dir, results)
                total_csvs += 1
                total_kernels += len(results)
            else:
                print(f"  Warning: No valid results in {case_name}/{powercap_name}")

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  CSV files created: {total_csvs}")
    print(f"  Total kernel results: {total_kernels}")
    print(f"{'='*60}")

    # Show example
    if total_csvs > 0:
        print("\nExample CSV format:")
        print("  id,perf(GFLOP/s),energy(mJ)")
        print("  1,123,12.35")
        print("  2,456,23.45")
        print("  ...")


if __name__ == '__main__':
    main()
