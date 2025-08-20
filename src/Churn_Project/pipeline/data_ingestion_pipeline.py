import os
import sys

root_path = os.path.abspath(os.path.join(os.getcwd()))
if root_path not in sys.path:
    sys.path.append(root_path)
print(root_path)

from src.Churn_Project.components.data_ingestion import DataIngestion
from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager
from src.Churn_Project.logging.logger import logging

try :
    logging.info(f">>>> Data ingestion started....")
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion =  DataIngestion(config = data_ingestion_config )
    data_ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed successfully")
    logging.info(f">>>> Data ingestion completed succesfully....")
except Exception as e :
    logging.error(f'Impossible de poursuivre {e}')
    raise e

