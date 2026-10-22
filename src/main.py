import yaml
import sys
import subprocess
import time
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError


## YAML structure defined with pydantic
class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="allow", ## make this forbid later
        strict=True
    )

class ModelConfig(StrictModel):
    num_keys: int = Field(gt=0)
    value_width: int = Field(ge=1)

class WorkersConfig(StrictModel):
    count: int = Field(gt=0)
    iterations: int = Field(gt=0)

class AccessConfig(StrictModel):
    mode: Literal["sparse", "dense"]
    keys_per_iter: int = Field(gt=0)

class ExperimentConfig(StrictModel):
    seed: int = 0
    model: ModelConfig
    workers: WorkersConfig
    access: AccessConfig

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
    p = subprocess.Popen(["python", "src/server.py", "--num-keys", str(config.model.num_keys), "--value-width", str(config.model.value_width), "--seed", str(config.seed)])

    ## waiting for server to go live
    time.sleep(0.5)
    
    ## launch workers
    clients = []
    for i in range(int(config.workers.count)):
        ## seed needs to be different on each worker but deterministically
        c = subprocess.Popen(["python", "src/client.py", "--num-keys", str(config.model.num_keys), "--value-width", str(config.model.value_width), "--seed", str(config.seed+i), "--mode", str(config.access.mode), "--keys-per-iter", str(config.access.keys_per_iter), "--iterations", str(config.workers.iterations)])
        clients.append(c)

    try:
        while p.poll() is None:
            time.sleep(1)
    finally:
        ## cleanup during development
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