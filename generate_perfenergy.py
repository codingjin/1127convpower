#!/usr/bin/env python3
"""
Complete aggregation and analysis pipeline for kernel performance and energy data.

This script performs three main tasks in sequence:
1. Parse output_kernel*.txt files and generate powercap{N}/results.csv
2. Generate perfenergy.csv and all.csv from results.csv files
3. Generate top1.csv and average.csv from perfenergy.csv

Output files per case:
- powercap1-5/results.csv - Raw performance and energy data per kernel
- all.csv - Combined raw data from all power caps
- perfenergy.csv - Performance loss and energy savings relative to powercap5
- top1.csv - Metrics for kernel id=1 (best kernel) formatted for plotting
- average.csv - Average metrics across all kernels formatted for plotting

Usage:
  python3 generate_perfenergy.py           # Process all cases
  python3 generate_perfenergy.py case1     # Process specific case only
"""

import os
import re
import csv
import sys
from pathlib import Path
from collections import defaultdict


# ============================================================================
# STEP 1: Parse output files and generate results.csv
# ============================================================================

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
            print(f"  Warning: Could not parse {output_file}")

    # Sort by kernel_id
    results.sort(key=lambda x: x[0])

    return results


def write_results_csv(powercap_dir, results):
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

    print(f"  ✓ Created {csv_path} ({len(results)} kernels)")


# ============================================================================
# STEP 2: Generate perfenergy.csv and all.csv
# ============================================================================

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
        print(f"  Error reading {filepath}: {e}")
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
            print(f"  Warning: kernel {kernel_id} not found in power cap data")
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

    print(f"  ✓ Generated: {output_file}")
    return output_file


def generate_perfenergy_csv(case_path, powercap_data, kernel_ids):
    """
    Generate perfenergy.csv showing performance loss and energy savings.

    Args:
        case_path: Path to case directory
        powercap_data: dict mapping powercap number -> kernel data
        kernel_ids: sorted list of kernel IDs
    """
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
    return perfenergy_file


# ============================================================================
# STEP 3: Generate top1.csv and average.csv
# ============================================================================

def get_power_caps(case_dir):
    """
    Auto-detect GPU type and return corresponding power cap values.

    Args:
        case_dir: Path to case directory

    Returns:
        list: Power cap values in watts [pc1, pc2, pc3, pc4] (excluding pc5)
    """
    # GPU power cap configurations (excluding powercap5)
    GPU_POWER_CAPS = {
        'RTX 3090': [100, 200, 300, 420],
        'RTX 4090': [150, 200, 300, 400],
        'A30': [100, 120, 140, 160],
        'A100': [100, 200, 300, 400],
    }

    # Try to detect GPU from output files
    try:
        output_file = next((case_dir / 'powercap1').glob('output_kernel*.txt'))
        with open(output_file, 'r') as f:
            content = f.read()
            gpu_match = re.search(r'GPU:\s+NVIDIA\s+(.+)', content)
            if gpu_match:
                gpu_name = gpu_match.group(1).strip()
                # Match GPU name to config
                for gpu_key, power_caps in GPU_POWER_CAPS.items():
                    if gpu_key in gpu_name:
                        return power_caps
    except Exception as e:
        print(f"  Warning: Could not detect GPU type: {e}")

    # Default to RTX 3090 if detection fails
    print(f"  Warning: Using default power caps (RTX 3090)")
    return [100, 200, 300, 420]


def generate_top1_csv(case_dir):
    """
    Generate top1.csv with performance/energy data for kernel id=1.

    Args:
        case_dir: Path to case directory
    """
    perfenergy_file = case_dir / 'perfenergy.csv'

    if not perfenergy_file.exists():
        print(f"  Warning: {perfenergy_file} not found, skipping top1.csv generation")
        return

    # Read perfenergy.csv and extract row for kernel id=1
    with open(perfenergy_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        rows = list(reader)

        if not rows:
            print(f"  Warning: No data in {perfenergy_file}")
            return

        # Get first row (kernel id=1), skip id column (index 0)
        first_row = rows[0][1:]  # Skip id column

    # Get power cap values
    power_caps = get_power_caps(case_dir)

    # Generate top1.csv
    top1_file = case_dir / 'top1.csv'

    with open(top1_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Performance Loss (%)', 'Energy Saved (%)', 'Power Cap (W)'])

        # Extract data for each power cap (columns are in pairs)
        for i in range(4):  # powercap 1-4
            perf_loss = first_row[i * 2]  # perf-loss(%)
            energy_saved = first_row[i * 2 + 1]  # energy-saved(%)
            power_cap = power_caps[i]
            writer.writerow([perf_loss, energy_saved, power_cap])

    print(f"  ✓ Generated: {top1_file}")


def generate_average_csv(case_dir):
    """
    Generate average.csv with average performance/energy across all kernels.

    Args:
        case_dir: Path to case directory
    """
    perfenergy_file = case_dir / 'perfenergy.csv'

    if not perfenergy_file.exists():
        print(f"  Warning: {perfenergy_file} not found, skipping average.csv generation")
        return

    # Read perfenergy.csv and calculate averages
    with open(perfenergy_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        rows = list(reader)

        if not rows:
            print(f"  Warning: No data in {perfenergy_file}")
            return

    # Calculate averages for each column (8 columns total: 4 power caps × 2 metrics)
    num_kernels = len(rows)

    # Initialize sums for each column
    sums = [0.0] * 8  # 4 powercaps × (perf-loss + energy-saved)

    for row in rows:
        # Skip first column (id), process columns 1-8
        for i in range(8):
            sums[i] += float(row[i + 1])

    # Calculate averages
    averages = [s / num_kernels for s in sums]

    # Get power cap values
    power_caps = get_power_caps(case_dir)

    # Generate average.csv
    average_file = case_dir / 'average.csv'

    with open(average_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Performance Loss (%)', 'Energy Saved (%)', 'Power Cap (W)'])

        # Write averages for each power cap
        for i in range(4):  # powercap 1-4
            perf_loss_avg = averages[i * 2]  # avg perf-loss(%)
            energy_saved_avg = averages[i * 2 + 1]  # avg energy-saved(%)
            power_cap = power_caps[i]
            writer.writerow([f'{perf_loss_avg:.2f}', f'{energy_saved_avg:.2f}', power_cap])

    print(f"  ✓ Generated: {average_file}")


# ============================================================================
# Main processing function
# ============================================================================

def process_case(case_dir):
    """
    Complete processing pipeline for a single case directory.

    Args:
        case_dir: Path to case directory

    Returns:
        tuple: (num_csvs_created, num_kernels_processed)
    """
    case_name = case_dir.name
    print(f"\nProcessing {case_name}/")
    print("=" * 60)

    num_csvs = 0
    num_kernels = 0

    # ========================================================================
    # STEP 1: Parse output files and generate results.csv for each power cap
    # ========================================================================
    print(f"\n[Step 1/3] Parsing output files and generating results.csv...")

    # Find all powercap directories
    powercap_dirs = sorted(case_dir.glob('powercap*'))

    if not powercap_dirs:
        print(f"  Warning: No powercap directories found in {case_name}")
        return 0, 0

    for powercap_dir in powercap_dirs:
        powercap_name = powercap_dir.name

        # Aggregate results for this powercap
        results = aggregate_powercap_directory(powercap_dir)

        if results:
            # Write CSV
            write_results_csv(powercap_dir, results)
            num_csvs += 1
            num_kernels = max(num_kernels, len(results))
        else:
            print(f"  Warning: No valid results in {case_name}/{powercap_name}")

    # ========================================================================
    # STEP 2: Generate perfenergy.csv and all.csv
    # ========================================================================
    print(f"\n[Step 2/3] Generating perfenergy.csv and all.csv...")

    # Read data from all 5 power caps
    powercap_data = {}
    for pc in range(1, 6):
        results_file = case_dir / f"powercap{pc}" / "results.csv"
        if not results_file.exists():
            print(f"  Warning: {results_file} not found")
            return num_csvs, num_kernels
        powercap_data[pc] = read_results_csv(results_file)

    # Check if powercap5 has data
    if not powercap_data[5]:
        print(f"  Error: No data in powercap5 for {case_name}")
        return num_csvs, num_kernels

    # Get all kernel IDs from powercap5 (reference)
    kernel_ids = sorted(powercap_data[5].keys())

    # Generate all.csv
    generate_all_csv(case_dir, powercap_data, kernel_ids)

    # Generate perfenergy.csv
    generate_perfenergy_csv(case_dir, powercap_data, kernel_ids)

    # ========================================================================
    # STEP 3: Generate top1.csv and average.csv
    # ========================================================================
    print(f"\n[Step 3/3] Generating top1.csv and average.csv...")

    generate_top1_csv(case_dir)
    generate_average_csv(case_dir)

    print(f"\n✓ Completed processing {case_name}")
    print(f"  - {num_kernels} kernels × 5 power caps")
    print(f"  - {num_csvs} results.csv files + 4 summary files")

    return num_csvs, num_kernels


def main():
    """Main function to process all cases or a specific case."""
    print("=" * 80)
    print("COMPLETE PERFORMANCE AND ENERGY DATA AGGREGATION PIPELINE")
    print("=" * 80)

    # Check if specific case_id provided as command-line argument
    target_case_id = sys.argv[1] if len(sys.argv) > 1 else None

    kernel_outputs_dir = Path('kernel_outputs')

    if not kernel_outputs_dir.exists():
        print(f"\nError: {kernel_outputs_dir} directory not found")
        print("Please run from the project root directory after running benchmarks")
        return

    # Find all case directories (or specific case if provided)
    if target_case_id:
        case_dirs = [kernel_outputs_dir / target_case_id]
        if not case_dirs[0].exists():
            print(f"\nError: {case_dirs[0]} directory not found")
            return
        print(f"\nProcessing specific case: {target_case_id}")
    else:
        case_dirs = sorted(kernel_outputs_dir.glob('case*'))
        if not case_dirs:
            print(f"\nError: No case directories found in {kernel_outputs_dir}")
            return
        print(f"\nFound {len(case_dirs)} case directories")

    # Process each case directory
    total_csvs = 0
    total_kernels = 0

    for case_dir in case_dirs:
        num_csvs, num_kernels = process_case(case_dir)
        total_csvs += num_csvs
        total_kernels = max(total_kernels, num_kernels)

    # Print final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Cases processed: {len(case_dirs)}")
    print(f"  Total results.csv files: {total_csvs}")
    print(f"  Max kernels per case: {total_kernels}")

    print(f"\nGenerated files per case:")
    print(f"  - powercap1-5/results.csv (raw performance and energy data)")
    print(f"  - all.csv (combined raw data from all power caps)")
    print(f"  - perfenergy.csv (performance loss and energy savings vs powercap5)")
    print(f"  - top1.csv (kernel id=1 metrics for plotting)")
    print(f"  - average.csv (average metrics across all kernels for plotting)")

    if case_dirs:
        print(f"\nExample location:")
        print(f"  {case_dirs[0]}/")
        print(f"    ├── powercap1/results.csv")
        print(f"    ├── powercap2/results.csv")
        print(f"    ├── powercap3/results.csv")
        print(f"    ├── powercap4/results.csv")
        print(f"    ├── powercap5/results.csv")
        print(f"    ├── all.csv")
        print(f"    ├── perfenergy.csv")
        print(f"    ├── top1.csv")
        print(f"    └── average.csv")
    print()


if __name__ == "__main__":
    main()
