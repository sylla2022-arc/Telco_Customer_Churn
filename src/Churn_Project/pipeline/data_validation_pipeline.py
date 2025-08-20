import os
import sys

root_path = os.path.abspath(os.path.join(os.getcwd()))
if root_path not in sys.path:
    sys.path.append(root_path)
print(root_path)
from src.Churn_Project.components.data_validation import DataValidation
from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager
from src.Churn_Project.logging.logger import logging

try :
    logging.info(f">>>> Data validation started....")
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation =  DataValidation(config = data_validation_config )
    data_validation.initiate_data_validation()
    print(f"Data validation completed successfully")
    logging.info(f">>>> Data validation completed succesfully....")
except Exception as e :
    logging.error(f'Impossible de poursuivre {e}')
    raise e

