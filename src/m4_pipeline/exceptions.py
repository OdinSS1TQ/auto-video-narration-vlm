"""
Exceptions — Custom error types for the pipeline.
"""


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class StepError(PipelineError):
    """Error in a specific pipeline step."""

    def __init__(self, step_name: str, message: str):
        self.step_name = step_name
        super().__init__(f"[{step_name}] {message}")


class VLMError(PipelineError):
    """Error from VLM processing."""
    pass


class TTSError(PipelineError):
    """Error from TTS processing."""
    pass


class SyncError(PipelineError):
    """Error from audio synchronization."""
    pass


class ConfigError(PipelineError):
    """Error in configuration."""
    pass


class ValidationError(PipelineError):
    """Error in data validation."""
    pass
