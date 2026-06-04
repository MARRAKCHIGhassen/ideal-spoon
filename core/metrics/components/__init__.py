# -*- coding: utf-8 -*-
"""
Metric Components Package.

This package contains atomic metric implementations. Unlike the high-level 
suites in the parent package (which aggregate multiple metrics), these modules 
perform the specific calculations for individual performance indicators.

Exports:
    * Dice: Computes the Dice Coefficient (F1 Score) for segmentation overlap.
    * ThroughputMetric: measures system processing speed (samples/sec) and latency.
"""

# -------------------------------------------------------------------------------------------------------------------

from .dice import Dice
from .throuput_metric import ThroughputMetric

# --