"""
Statistical analysis and H₁ validation.
"""
import pandas as pd
import numpy as np

def analyze(input_path: str, report_path: str):
    df = pd.read_csv(input_path)
    
    # Aggregate by generator type
    agg = df.groupby('generator').agg({
        'enrichment_ratio': ['mean', 'std'],
        'ks_pvalue': 'median',
        'mw_pvalue': 'median', 
        'checks_to_find_factor': 'mean',
        'score_variance': 'mean'
    }).round(6)
    
    # Compute check reduction: asymmetric_qmc vs symmetric_random
    baseline_checks = df[df['generator'] == 'symmetric_random']['checks_to_find_factor'].mean()
    asymmetric_checks = df[df['generator'] == 'asymmetric_qmc']['checks_to_find_factor'].mean()
    check_reduction = (baseline_checks - asymmetric_checks) / baseline_checks * 100 if baseline_checks > 0 else 0
    
    # Variance ratio: QMC vs PRN
    prn_var = df[df['generator'] == 'symmetric_random']['score_variance'].mean()
    qmc_var = df[df['generator'] == 'asymmetric_qmc']['score_variance'].mean()
    variance_ratio = qmc_var / prn_var if prn_var > 0 else float('inf')
    
    # H₁ validation
    asym_df = df[df['generator'] == 'asymmetric_qmc']
    h1_enrichment = asym_df['enrichment_ratio'].mean() >= 4.0
    h1_ks = asym_df['ks_pvalue'].median() < 1e-10
    h1_reduction = 30 <= check_reduction <= 70
    h1_variance = variance_ratio < 0.5
    
    report = f"""# Validation Report

## Aggregate Metrics
{agg.to_markdown()}

## H₁ Success Criteria
| Metric | Threshold | Observed | Pass |
|--------|-----------|----------|------|
| Q-enrichment | ≥4.0× | {asym_df['enrichment_ratio'].mean():.2f}× | {'✓' if h1_enrichment else '✗'} |
| KS p-value | <1e-10 | {asym_df['ks_pvalue'].median():.2e} | {'✓' if h1_ks else '✗'} |
| Check reduction | 30-70% | {check_reduction:.1f}% | {'✓' if h1_reduction else '✗'} |
| Variance ratio | <0.5 | {variance_ratio:.3f} | {'✓' if h1_variance else '✗'} |

## Conclusion
**{'H₁ SUPPORTED' if all([h1_enrichment, h1_ks, h1_reduction, h1_variance]) else 'H₀ NOT REJECTED'}**
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    print(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results.csv")
    parser.add_argument("--report", default="validation_report.md")
    args = parser.parse_args()
    analyze(args.input, args.report)
