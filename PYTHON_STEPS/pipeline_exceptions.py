"""
Custom exception hierarchy for CFD simulation pipeline.
Provides domain-specific errors for each pipeline stage with contextual information.
"""

class PipelineStepError(Exception):
    """Base exception for all pipeline step failures"""
    def __init__(self, message: str, step_name: str, details: dict = {}):
        self.step_name = step_name
        self.details = details if details else {}
        super().__init__(message)
    
    def to_dict(self):
        """Convert exception to dict for DB storage"""
        return {
            'step': self.step_name,
            'message': str(self),
            'details': self.details
        }


class GeometryStepError(PipelineStepError):
    """Raised when geometry generation fails (Step 1: JSON → Geometry)"""
    def __init__(self, message: str, details: dict = {}):
        super().__init__(message, 'geometry', details if details else {})


class MeshingStepError(PipelineStepError):
    """Raised when mesh generation fails (Step 2: Geometry → Mesh)"""
    def __init__(self, message: str, details: dict = {}):
        super().__init__(message, 'meshing', details if details else {})


class CFDSetupError(PipelineStepError):
    """Raised when CFD case setup fails (Step 3: Mesh → CFD)"""
    def __init__(self, message: str, details: dict = {}):
        super().__init__(message, 'cfd_setup', details if details else {})


class SubmissionError(PipelineStepError):
    """Raised when cloud submission fails (Step 4: Submit to Inductiva)"""
    def __init__(self, message: str, details: dict = {}):
        super().__init__(message, 'cloud_execution', details if details else {})
