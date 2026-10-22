import yaml
import sys
import subprocess
import time

def yaml_processor(path):
    with open(path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Yaml processing error\n", exc)
    return None

## needs better error handling possibly
def process_launcher(config):
    seed = config.get("seed", 0)
    num_keys = config.get("model", {}).get("num_keys", None)
    value_width = config.get("model", {}).get("value_width", None)

    ## launch server
    p = subprocess.Popen(["python", "src/server.py", "--num-keys", str(num_keys), "--value-width", str(value_width), "--seed", str(seed)])

    while p.poll() is None:
        time.sleep(1)

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