"""
Simulation entry point: parses YAML and triggers the harness driver.
"""
import sys
from src.harness.config import load_config
from src.harness.driver import Driver


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)
    if not config:
        sys.exit(1)

    driver = Driver(config)
    driver.run()


if __name__ == "__main__":
    main()