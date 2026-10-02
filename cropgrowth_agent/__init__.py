"""Rice growth-stage agent: natural-language requests → growth-stage maps from satellite NDVI."""
from . import data_processing, phenology, pipeline
from .runner import RunConfig, Runner

__all__ = ["data_processing", "phenology", "pipeline", "RunConfig", "Runner", "CropGrowthAgent"]


def __getattr__(name):
    if name == "CropGrowthAgent":     # lazy: the pipeline works without the google-genai package
        from .agent import CropGrowthAgent
        return CropGrowthAgent
    raise AttributeError(name)
