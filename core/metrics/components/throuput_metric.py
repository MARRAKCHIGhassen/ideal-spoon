# -*- coding: utf-8 -*-
"""
Phase‑Aware Scattering‑Inspired Encoder–Decoders: Throughput Metric
==================================================================

This module implements a performance profiling metric used to measure 
system throughput and latency. It tracks the number of samples processed 
over time to provide insights into model efficiency and hardware utilization.
"""

# -------------------------------------------------------------------------------------------------------------------

# Importing Global Libraries
from __future__ import annotations

import logging

import torch
from torchmetrics import Metric
from typing import Final, Callable, Any, Dict

# Import custom libraries

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# LOGGERS
# ---------------------------------------------------------------------

debug_logger = logging.getLogger(f"{__name__.split('.')[0]}.debug")
semantic_logger = logging.getLogger(f"{__name__.split('.')[0]}.semantic")

# -------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------
# IMPLEMENTATION
# ---------------------------------------------------------------------

# == ThroughputMetric ==
class ThroughputMetric(Metric):
    """
    Metric to calculate processing throughput and latency.

    Tracks:
    1. **Throughput**: Samples per second (higher is better).
    2. **Latency**: Average milliseconds per batch (lower is better).

    Attributes
    ----------
    total_samples : torch.Tensor
        Total number of items processed.
    total_time : torch.Tensor
        Total accumulated inference time in seconds.
    batch_count : torch.Tensor
        Number of batches processed.
    """
    
    # ===========

    total_samples: torch.Tensor
    total_time   : torch.Tensor
    batch_count  : torch.Tensor

    # -- __init__
    def __init__(self):
        """
        Initializes the throughput metric states.
        """
        # ------------------------------------------------

        super().__init__()
        self.add_state("total_samples", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("batch_count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        debug_logger.debug("ThroughputMetric state initialized.")

    # -- update
    def update(self, batch_size: int, duration_seconds: float):
        """
        Updates the metric states with new timing data.

        Parameters
        ----------
        batch_size : int
            The number of samples in the current batch.
        duration_seconds : float
            The time taken to process the batch.
        """
        # ------------------------------------------------

        debug_logger.debug(f"Updating metric: batch_size={batch_size}, duration={duration_seconds:.4f}s")
        
        self.total_samples += batch_size
        self.total_time += duration_seconds
        self.batch_count += 1

    # -- compute
    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Calculates final throughput and latency statistics.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dictionary containing:
            - 'throughput': Samples per second.
            - 'latency_ms': Average time per batch in milliseconds.
        """
        # ------------------------------------------------

        debug_logger.debug("Computing throughput and latency statistics.")

        if self.total_time == 0:
            debug_logger.warning("Total time is zero; returning null metrics.")
            return {"throughput": torch.tensor(0.0), "latency_ms": torch.tensor(0.0)}
            
        throughput = self.total_samples / self.total_time
        latency = (self.total_time / self.batch_count) * 1000.0
        
        debug_logger.debug(f"Computed stats - Throughput: {throughput.item():.2f}, Latency: {latency.item():.2f}ms")

        return {
            "throughput": throughput,
            "latency_ms": latency
        }
