from src.Churn_Project.components.model_trainer import ModelTrainer
from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager
from src.Churn_Project.configuation_manager.config_manager import ConfigurationManager as ConfigurationManager_trans
from src.Churn_Project.logging.logger import logging

try :
    logging.info(f">>>> Model trainer start....")
    # initialiser la config
    config = ConfigurationManager()
    # récupérer data transfo config
    config_trans = ConfigurationManager_trans()
    data_transformation_config = config_trans.get_data_transformation_config()  
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

