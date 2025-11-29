#!/usr/bin/env python3
"""
Generate perfenergy.csv and all.csv files for each case.

perfenergy.csv: Shows performance loss and energy savings relative to powercap5
- perf-loss(%) = ((perf_pc5 - perf_pcN) / perf_pc5) * 100
- energy-saved(%) = ((energy_pc5 - energy_pcN) / energy_pc5) * 100

all.csv: Combines all powercap1-5 raw performance and energy data

Output:
- kernel_outputs/case{X}/perfenergy.csv
- kernel_outputs/case{X}/all.csv
"""

import os
import csv
from pathlib import Path
from collections import defaultdict


def read_results_csv(filepath):
    """
    Read a results.csv file and return a dictionary mapping id -> (perf, energy).

    Returns:
        dict: {id: (perf_gflops, energy_mj)}
    """
    data = {}
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kernel_id = int(row['id'])
                perf = float(row['perf(GFLOP/s)'])
                energy = float(row['energy(mJ)'])
                data[kernel_id] = (perf, energy)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    return data


def calculate_metrics(powercap_data, powercap5_data):
    """
    Calculate perf-loss(%) and energy-saved(%) for a power cap relative to powercap5.

    Args:
        powercap_data: dict mapping id -> (perf, energy) for the power cap
        powercap5_data: dict mapping id -> (perf, energy) for powercap5

    Returns:
        dict: {id: (perf_loss_pct, energy_saved_pct)}
    """
    metrics = {}

    for kernel_id in powercap5_data.keys():
        if kernel_id not in powercap_data:
            print(f"Warning: kernel {kernel_id} not found in power cap data")
            continue

        perf_pc5, energy_pc5 = powercap5_data[kernel_id]
        perf_pcN, energy_pcN = powercap_data[kernel_id]

        # perf-loss(%) = ((perf_pc5 - perf_pcN) / perf_pc5) * 100
        # Higher is worse (more performance loss)
        if perf_pc5 > 0:
            perf_loss_pct = ((perf_pc5 - perf_pcN) / perf_pc5) * 100
        else:
            perf_loss_pct = 0.0

        # energy-saved(%) = ((energy_pc5 - energy_pcN) / energy_pc5) * 100
        # Higher is better (more energy saved)
        if energy_pc5 > 0:
            energy_saved_pct = ((energy_pc5 - energy_pcN) / energy_pc5) * 100
        else:
            energy_saved_pct = 0.0

        metrics[kernel_id] = (perf_loss_pct, energy_saved_pct)

    return metrics


def generate_all_csv(case_path, powercap_data, kernel_ids):
    """
    Generate all.csv combining all powercap data.

    Args:
        case_path: Path to case directory
        powercap_data: dict mapping powercap number -> kernel data
        kernel_ids: sorted list of kernel IDs
    """
    output_file = case_path / "all.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header (simplified - no powercap prefixes)
        header = ['id']
        for pc in range(1, 6):
            header.append('perf(GFLOP/s)')
            header.append('energy(mJ)')
        writer.writerow(header)

        # Write data rows
        for kernel_id in kernel_ids:
            row = [kernel_id]
            for pc in range(1, 6):
                if kernel_id in powercap_data[pc]:
                    perf, energy = powercap_data[pc][kernel_id]
                    row.append(f"{perf:.0f}")  # Round to integer
                    row.append(f"{energy:.2f}")  # 2 decimal places
                else:
                    row.append("N/A")
                    row.append("N/A")
            writer.writerow(row)

    return output_file


def generate_perfenergy_csv(case_dir):
    """
    Generate perfenergy.csv for a single case directory.

    Args:
        case_dir: Path to case directory (e.g., kernel_outputs/case1)
    """
    case_path = Path(case_dir)
    case_name = case_path.name

    print(f"\nProcessing {case_name}...")

    # Read data from all 5 power caps
    powercap_data = {}
    for pc in range(1, 6):
        results_file = case_path / f"powercap{pc}" / "results.csv"
        if not results_file.exists():
            print(f"  Warning: {results_file} not found, skipping {case_name}")
            return
        powercap_data[pc] = read_results_csv(results_file)

    # Check if powercap5 has data
    if not powercap_data[5]:
        print(f"  Error: No data in powercap5 for {case_name}")
        return

    # Get all kernel IDs from powercap5 (reference)
    kernel_ids = sorted(powercap_data[5].keys())
    num_kernels = len(kernel_ids)
    print(f"  Found {num_kernels} kernels in {case_name}")

    # Generate all.csv (combines all powercap data)
    all_csv_file = generate_all_csv(case_path, powercap_data, kernel_ids)
    print(f"  ✓ Generated: {all_csv_file}")
    print(f"  ✓ {num_kernels} kernels × 5 power caps")

    # Calculate metrics for powercap 1-4 relative to powercap5
    metrics = {}
    for pc in range(1, 5):
        metrics[pc] = calculate_metrics(powercap_data[pc], powercap_data[5])

    # Write perfenergy.csv
    perfenergy_file = case_path / "perfenergy.csv"
    with open(perfenergy_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header (simplified - no powercap prefixes)
        header = ['id']
        for pc in range(1, 5):
            header.append('perf-loss(%)')
            header.append('energy-saved(%)')
        writer.writerow(header)

        # Write data rows
        for kernel_id in kernel_ids:
            row = [kernel_id]
            for pc in range(1, 5):
                if kernel_id in metrics[pc]:
                    perf_loss, energy_saved = metrics[pc][kernel_id]
                    row.append(f"{perf_loss:.2f}")
                    row.append(f"{energy_saved:.2f}")
                else:
                    row.append("N/A")
                    row.append("N/A")
            writer.writerow(row)

    print(f"  ✓ Generated: {perfenergy_file}")
    print(f"  ✓ {num_kernels} kernels × 4 power caps (relative to powercap5)")


def main():
    """Main function to process all cases."""
    print("="*80)
    print("Generating perfenergy.csv and all.csv files for all cases")
    print("="*80)

    kernel_outputs_dir = Path("kernel_outputs")

    if not kernel_outputs_dir.exists():
        print(f"Error: {kernel_outputs_dir} directory not found!")
        print("Make sure you've run the benchmarks first.")
        return

    # Find all case directories
    case_dirs = sorted([d for d in kernel_outputs_dir.iterdir()
                       if d.is_dir() and d.name.startswith('case')])

    if not case_dirs:
        print(f"Error: No case directories found in {kernel_outputs_dir}")
        return

    print(f"Found {len(case_dirs)} case directories")

    # Process each case
    for case_dir in case_dirs:
        generate_perfenergy_csv(case_dir)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Generated files in each case directory:")
    for case_dir in case_dirs:
        print(f"  - {case_dir}/all.csv")
        print(f"  - {case_dir}/perfenergy.csv")
    print()


if __name__ == "__main__":
    main()
