"""Run training & evaluation against examples/config.yaml."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlexplore.cli import main

if __name__ == "__main__":
    sys.argv = ["rlexplore", "--config", "examples/config.yaml", "--mode", "both"]
    main()
