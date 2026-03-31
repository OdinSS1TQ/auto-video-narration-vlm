"""
Pipeline Module — Orchestrator for the end-to-end dubbing pipeline.
"""

from src.m4_pipeline.runner import PipelineRunner
from src.m4_pipeline.config import PipelineConfig

__all__ = ["PipelineRunner", "PipelineConfig"]
