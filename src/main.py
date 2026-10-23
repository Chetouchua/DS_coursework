import yaml
import sys
import os
import subprocess
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError
import numpy as np

## YAML structure defined with pydantic
## Not all fields defined , only those needed for implementation so far, add as you need!
class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="allow", ## make this forbid later
        strict=True
    )

class ModelTargetConfig(StrictModel):
    dist: Literal["normal"]
    mean: float
    std: float = Field(ge=0)

class ModelConfig(StrictModel):
    num_keys: int = Field(gt=0)
    value_width: int = Field(ge=1)
    learning_rate: float
    target: ModelTargetConfig

class WorkersConfig(StrictModel):
    count: int = Field(gt=0)
    iterations: int = Field(gt=0)

class AccessConfig(StrictModel):
    mode: Literal["sparse", "dense"]
    keys_per_iter: int = Field(gt=0)

class GradientConfig(StrictModel):
    type: Literal["target_diff", "fixed"]
    noise_std: float

class ExperimentConfig(StrictModel):
    experiment: str = Field(pattern=r'^[a-zA-Z_]+$')
    seed: int = 0
    model: ModelConfig
    workers: WorkersConfig
    access: AccessConfig
    gradient: GradientConfig

## Basic Parser
def yaml_processor(path) -> ExperimentConfig:
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Yaml processing error\n", exc)
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as e:
        print("Invalid experiment configuration:")
        print(e)
        return None

## Subprocess launcher (needs to be made compatible with all required features)
def process_launcher(config: ExperimentConfig):
    ## launch server
    p = subprocess.Popen(["python", "src/server.py", "--num-keys", str(config.model.num_keys), "--value-width", str(config.model.value_width), "--seed", str(config.seed), "--learning-rate", str(config.model.learning_rate)])

    ## waiting for server to go live
    time.sleep(0.25)

    ## Make & save target weights
    if not os.path.isdir("data"):
        os.mkdir("data")
    if config.seed:
        rng = np.random.default_rng(config.seed)
    else:
        rng = np.random.default_rng(None)

    target = rng.normal(
        loc=config.model.target.mean,
        scale=config.model.target.std,
        size=(config.model.num_keys, config.model.value_width)
    )
    file_name = f"data/{config.experiment}.npy"
    np.save(file_name, target)

    ## launch workers
    clients = []
    for i in range(int(config.workers.count)):
        ## seed needs to be different on each worker but deterministically
        c = subprocess.Popen(["python", "src/client.py", "--num-keys", str(config.model.num_keys), "--value-width", str(config.model.value_width), "--seed", str(config.seed+i), "--mode", str(config.access.mode), "--keys-per-iter", str(config.access.keys_per_iter), "--iterations", str(config.workers.iterations), "--target-weight-file", file_name, "--gradient-type", config.gradient.type, "--gradient-noise", str(config.gradient.noise_std)])
        clients.append(c)

    try:
        while p.poll() is None:
            time.sleep(1)
    finally:
        ## cleanup during development
        print("\n\ntarget is", target, "\n")
        print("CTRL+C Received, Ensuring processes are killed, please wait")
        p.kill()
        p.wait()
        for c in clients:
            c.kill()
            c.wait()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Enter exactly one argument specifying the yaml file path eg: python main.py test.yaml")
        exit(0)
    yaml_path = sys.argv[1]
    sim_config = yaml_processor(yaml_path)
    if sim_config:
        process_launcher(sim_config)
        exit(1)
    else:
        exit(0)