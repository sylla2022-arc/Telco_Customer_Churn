import os
import sys

from src.Churn_Project.components.data_ingestion import DataIngestion
from src.Churn_Project.components.data_validation import DataValidation
from src.Churn_Project.components.data_transformation import DataTransformation

from src.Churn_Project.components.model_trainer import ModelTrainer
from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager as ConfigurationManager_trans

from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager
from src.Churn_Project.logging.logger import logging

logging.info("Test log: main.py started")
try :
    logging.info(f">>>> Data ingestion started....")
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion =  DataIngestion(config = data_ingestion_config )
    data_ingestion.initiate_data_ingestion()
    print(f">>>>>> Data ingestion completed successfully")
    logging.info(f">>>> Data ingestion completed succesfully....")
except Exception as e :
    logging.error(f'Impossible de poursuivre {e}')
    raise e

try :
    logging.info(f">>>> Data validation started....")
    config = ConfigurationManager()
    data_validation_config = config.get_data_validation_config()
    data_validation =  DataValidation(config = data_validation_config )
    data_validation.initiate_data_validation()
    print(f">>>>>>> Data validation completed successfully")
    logging.info(f">>>> Data validation completed succesfully....")
except Exception as e :
    logging.error(f'Impossible de poursuivre {e}')
    raise e

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

try :
    logging.info(f">>>> Model trainer start....")
    # initialiser la config
    config = ConfigurationManager()
    # récupérer data transfo config
    # config_trans = ConfigurationManager_trans()
    # data_transformation_config = config_trans.get_data_transformation_config()  
    model_trainer_config = config.get_model_trainer_config()
    model_trainer = ModelTrainer(
        config=model_trainer_config,
        data_transformation_config=data_transformation_config
    )
    # lancer l'entraînement
    model_trainer.initiate_model_trainer()
    print(f"Model trainer completed successfully")
    logging.info(f">>>> Model trainer completed succesfully....")
except Exception as e :
    logging.error(f'Impossible de poursuivre {e}')
    raise e
