# Validation Report

## Aggregate Metrics
| generator        |   ('enrichment_ratio', 'mean') |   ('enrichment_ratio', 'std') |   ('ks_pvalue', 'median') |   ('mw_pvalue', 'median') |   ('checks_to_find_factor', 'mean') |   ('score_variance', 'mean') |
|:-----------------|-------------------------------:|------------------------------:|--------------------------:|--------------------------:|------------------------------------:|-----------------------------:|
| asymmetric_qmc   |                            inf |                           nan |                         1 |                         0 |                                 236 |                            0 |
| symmetric_qmc    |                            inf |                           nan |                         1 |                         0 |                                 511 |                            0 |
| symmetric_random |                            inf |                           nan |                         1 |                         0 |                                 506 |                            0 |

## H₁ Success Criteria
| Metric | Threshold | Observed | Pass |
|--------|-----------|----------|------|
| Q-enrichment | ≥4.0× | inf× | ✓ |
| KS p-value | <1e-10 | 1.00e+00 | ✗ |
| Check reduction | 30-70% | 53.4% | ✓ |
| Variance ratio | <0.5 | 1.064 | ✗ |

## Conclusion
**H₀ NOT REJECTED**
