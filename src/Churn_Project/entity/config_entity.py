from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    """
    Configuration for data ingestion.
    """
    root_path: Path 
    source_dir: Path
    raw_file_path: Path 

    def __init__(self, root_dir, source_dir, raw_file_path):
        self.root_dir = root_dir
        self.source_dir = source_dir
        self.raw_file_path = raw_file_path

@dataclass
class DataValidationConfig:
    """
    Configuration for data validation.
    """
    root_dir: Path 
    data_dir: Path
    status_file: Path 
    feature : dict
    target : dict

    def __init__(self, root_dir, data_dir, status_file, feature, target):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.status_file= status_file
        self.feature = feature
        self.target = target