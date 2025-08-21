import os
import sys

from src.Churn_Project.components.data_transformation import DataTransformation
from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager
from src.Churn_Project.logging.logger import logging

try :
    logging.info(f">>>> Data transformation start....")
    config = ConfigurationManager()
    data_transformation_config = config.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.initiate_data_transformation()
    print(f"Data transformation completed successfully")
    logging.info(f">>>> Data transformation completed succesfully....")
except Exception as e :
    logging.error(f'Impossible de poursuivre {e}')
    raise e


