import os
import pandas as pd
from src.Churn_Project.entity.config_entity import DataIngestionConfig
from src.Churn_Project.logging.logger import logging
from src.Churn_Project.utils.utility import  create_directory


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def initiate_data_ingestion(self):
        logging.info('Data ingestion initiated')
        try:

            if not os.path.exists(self.config.source_dir):
                raise FileNotFoundError(f"Le fichier source est introuvable : {self.config.source_dir}")

            data = pd.read_csv(self.config.source_dir)
        except Exception as e:
            logging.error(f"Failed to read source data: {e}")
            raise e
        
        print(f"The data shape is {data.shape}")
        logging.info(f"Data loaded successfully from {self.config.source_dir}")
        logging.info(f"Data shape before dropna: {data.shape}")
        
        data.dropna(inplace=True)
        
        # Save raw
        try:
            create_directory([os.path.dirname(self.config.raw_file_path)]) 
            data.to_csv(self.config.raw_file_path, index=False, header=True)
            logging.info(f"Data saved successfully to {self.config.raw_file_path}")
        except Exception as e:
            logging.error(f"Failed to save raw data: {e}")
            raise e