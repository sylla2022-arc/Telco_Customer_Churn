import os
import yaml
import sys

print(os.getcwd())
root_path = os.path.abspath(os.path.join(os.getcwd()))
if root_path not in sys.path:
    sys.path.append(root_path)
print(root_path)
from src.Churn_Project.logging.logger import logging


current_wd = os.getcwd()
CONFIG_YAML_FILE = os.path.join('config' ,  "config.yaml")

