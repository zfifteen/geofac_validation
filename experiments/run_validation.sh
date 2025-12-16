#!/bin/bash
# Run Z5D Validation Workflow for N_127
# 
# This script runs the complete validation pipeline:
# 1. Generate and score candidates
# 2. Display summary
# 3. Optional: Open analysis notebook

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================================================"
echo "Z5D VALIDATION WORKFLOW FOR N_127"
echo "========================================================================"
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "import gmpy2, mpmath, pandas, numpy, scipy" 2>/dev/null || {
    echo "ERROR: Missing dependencies. Install with:"
    echo "  pip install gmpy2 mpmath pandas numpy scipy matplotlib seaborn"
    exit 1
}
echo "✓ All dependencies installed"
echo ""

# Step 1: Run experiment
echo "========================================================================"
echo "STEP 1: Running Validation Experiment"
echo "========================================================================"
echo ""
echo "Generating 100,000 candidates and scoring with Z5D..."
echo "Estimated time: ~10 seconds"
echo ""

python3 experiments/z5d_validation_n127.py

echo ""
echo "✓ Experiment complete!"
echo ""

# Step 2: Display summary
echo "========================================================================"
echo "STEP 2: Results Summary"
echo "========================================================================"
echo ""

python3 experiments/summarize_results.py

echo ""

# Step 3: Optional notebook
echo "========================================================================"
echo "NEXT STEPS"
echo "========================================================================"
echo ""
echo "1. Review detailed report:"
echo "   cat docs/z5d_validation_n127_results.md"
echo ""
echo "2. Analyze data with Jupyter notebook:"
echo "   jupyter notebook notebooks/z5d_validation_analysis.ipynb"
echo ""
echo "3. Scale to 1M candidates (edit experiments/z5d_validation_n127.py):"
echo "   Change NUM_CANDIDATES = 100_000 to NUM_CANDIDATES = 1_000_000"
echo "   Re-run: python3 experiments/z5d_validation_n127.py"
echo ""
echo "4. View raw data:"
echo "   head -20 data/z5d_validation_n127_results.csv"
echo ""
echo "========================================================================"
echo "WORKFLOW COMPLETE"
echo "========================================================================"
