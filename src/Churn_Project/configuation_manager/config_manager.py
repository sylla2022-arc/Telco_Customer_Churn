from src.Churn_Project.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
from src.Churn_Project.constants import CONFIG_YAML_FILE_PATH, COL_YAML_FILE_PATH
from src.Churn_Project.utils.utility import (read_yaml_file, create_directory)
from pathlib import Path
from src.Churn_Project.logging.logger import logging

class ConfigurationManager :
    def __init__(self,
        config_file_path : str = CONFIG_YAML_FILE_PATH,
        config_features_target_path : str = COL_YAML_FILE_PATH):
        self.config = read_yaml_file(config_file_path)
        config_feature_target = read_yaml_file(config_features_target_path)
        self.feature_target = config_feature_target

        print(f"Type of artifacts_root: {type(self.config.artifact_root)}") 
        print(f"Value of artifacts_root: {self.config.data_ingestion}" )

        create_directory([self.config.artifact_root])
        logging.info(f"Folder successfully crated to {self.config.artifact_root}")

    def get_data_ingestion_config(self)-> DataIngestionConfig  :
        config = self.config.data_ingestion

        create_directory([config.root_dir])
        logging.info(f"DI: Folder successfully crated to {config.root_dir}")
        data_ingestion_config = DataIngestionConfig(root_dir = config.root_dir,
        source_dir = config.source_dir,
        raw_file_path = config.raw_file_path)

        return data_ingestion_config

        
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        feature = self.feature_target.COLUMNS
        target = self.feature_target.TARGET

        root_dir = config.root_dir
        data_dir = config.data_dir
        status_file =  config.status_file
    
        create_directory([config.root_dir])
        logging.info(f"DV: Folder successfully crated to {config.root_dir}")
        
        data_validation_config = DataValidationConfig(
                            root_dir = Path(root_dir),
                        data_dir = Path(data_dir),
                        status_file = Path(status_file),
                        feature =  dict(feature),
                        target = dict(target)

        )
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directory([config.root_dir])

        root_dir = config.root_dir
        data_dir = config.data_dir
        train_file_path = config.train_file_path
        test_file_path = config.test_file_path
        y_train_array = config.y_train_array
        y_test_array = config.y_test_array
        preprocessor_obj = config.preprocessor_obj

        data_transformation_config = DataTransformationConfig(
            root_dir = Path(root_dir),
            data_dir = Path(data_dir),
            train_file_path= Path(train_file_path),
            test_file_path = Path(test_file_path),
            y_train_array = Path(y_train_array),
            y_test_array = Path(y_test_array),
            preprocessor_obj = Path(preprocessor_obj)
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig :
        config = self.config.model_trainer

        create_directory([config.root_dir])

        model_trainer = ModelTrainerConfig(
            root_dir= Path(config.root_dir),
            model_obj= Path(config.model_obj)
        )
        return model_trainer
