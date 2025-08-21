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

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    """
    root_dir: Path 
    data_dir: Path
    train_file_path: Path 
    test_file_path : Path
    y_train_array = Path
    y_test_array = Path
    preprocessor_obj : Path

    def __init__(self, root_dir, data_dir, train_file_path, test_file_path, 
                 y_train_array, y_test_array, preprocessor_obj):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.train_file_path= train_file_path
        self.test_file_path = test_file_path
        self.y_train_array = y_train_array
        self.y_test_array = y_test_array
        self.preprocessor_obj = preprocessor_obj