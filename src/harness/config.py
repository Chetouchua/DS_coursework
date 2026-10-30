"""
Pydantic configuration models for Parameter Server experiment benchmarks.
"""
from typing import Any, Dict, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ClusterConfig(BaseSchema):
    servers: int = 1
    replication: int = 0


class ConsistencyConfig(BaseSchema):
    mode: str = "sequential"
    tau: int = 0


class TargetConfig(BaseSchema):
    dist: str = "normal"
    mean: float = 0.0
    std: float = 1.0


class ModelConfig(BaseSchema):
    num_keys: int = 1000
    value_width: int = 1
    init: str = "zeros"
    learning_rate: float = 0.05
    target: Optional[TargetConfig] = None


class DistValueConfig(BaseSchema):
    dist: str = "constant"
    value: float = 5.0


class StragglersConfig(BaseSchema):
    fraction: float = 0.0
    slowdown: float = 1.0


class AccessConfig(BaseSchema):
    mode: str = "dense"
    range_size: Optional[int] = 1000
    keys_per_iter: Optional[int] = None
    alpha: Optional[float] = None


class GradientConfig(BaseSchema):
    type: str = "target_diff"
    noise_std: float = 0.0


class WorkersConfig(BaseSchema):
    count: int = 2
    iterations: int = 5
    compute_ms: Optional[DistValueConfig] = None
    stragglers: Optional[StragglersConfig] = None
    access: Optional[AccessConfig] = Field(default_factory=AccessConfig)
    gradient: Optional[GradientConfig] = Field(default_factory=GradientConfig)


class OutputConfig(BaseSchema):
    dir: str = "results/sanity_run"
    observer_every: int = 1


class ExperimentConfig(BaseSchema):
    experiment: str
    seed: int = 42
    cluster: Optional[ClusterConfig] = Field(default_factory=ClusterConfig)
    consistency: Optional[ConsistencyConfig] = Field(default_factory=ConsistencyConfig)
    model: ModelConfig
    workers: WorkersConfig
    output: Optional[OutputConfig] = Field(default_factory=OutputConfig)


def load_config(path: str) -> ExperimentConfig:
    """Parses and validates the YAML experiment configuration."""
    with open(path, "r") as stream:
        data = yaml.safe_load(stream)
    return ExperimentConfig(**data)