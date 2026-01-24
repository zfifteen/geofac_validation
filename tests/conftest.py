"""
pytest configuration for adaptive enrichment tests.
"""
import sys
from pathlib import Path

# Add experiments/adaptive_enrichment to path so tests can import modules
adaptive_enrichment_path = Path(__file__).parent.parent / "experiments" / "adaptive_enrichment"
sys.path.insert(0, str(adaptive_enrichment_path))
