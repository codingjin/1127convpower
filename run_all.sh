#!/bin/bash
# Master script to run all generated kernels × all power caps
# Total: 175 kernels × 5 power caps = 875 runs across 7 cases
# WARNING: This will run ALL kernels with ALL power caps sequentially!
# This may take a VERY long time. Consider using layer scripts instead.
# Recommended: Run layer scripts on different GPUs in parallel

echo "Running all 175 kernels × 5 power caps from 7 layers..."
echo "Total runs: 875"
echo "============================================================"


echo ""
echo "Running layer: case1 (all power caps)..."
bash run_layer_case1.sh

echo ""
echo "Running layer: case2 (all power caps)..."
bash run_layer_case2.sh

echo ""
echo "Running layer: case3 (all power caps)..."
bash run_layer_case3.sh

echo ""
echo "Running layer: case4 (all power caps)..."
bash run_layer_case4.sh

echo ""
echo "Running layer: case5 (all power caps)..."
bash run_layer_case5.sh

echo ""
echo "Running layer: case6 (all power caps)..."
bash run_layer_case6.sh

echo ""
echo "Running layer: case7 (all power caps)..."
bash run_layer_case7.sh

echo ""
echo "All kernels × power caps completed!"
echo "Total runs completed: 875"
echo ""
echo "============================================================"
echo "Aggregating results into CSV files..."
echo "============================================================"
python3 generate_perfenergy.py

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "Results aggregated to: kernel_outputs/<case_id>/powercap<N>/results.csv"
