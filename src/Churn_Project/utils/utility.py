import os
import yaml
from src.Churn_Project.logging.logger import logging
import sys
from box import ConfigBox #type:ignore
import joblib

def read_yaml_file(path_yml) -> ConfigBox :
    try:
        if not os.path.exists(path_yml):
            logging.error(f"Le fichier yaml {path_yml} n'existe pas.")
            raise FileNotFoundError(f"Le fichier yaml {path_yml} n'existe pas.")
        with open(path_yml, 'r') as f:
            content = yaml.safe_load(f)
            logging.info(f"yaml file successfully loaded from {path_yml}")
            return ConfigBox(content)
    except Exception as e:
        logging.error(f'Impossible de lire le fichier yaml: {e}')
        raise e


def create_directory(path_list: list,  verbose=True):
    for path in path_list:
        if not path:
            logging.error('Impossible de créer un répertoire')
            sys.exit(1)
        try:
            os.makedirs(path, exist_ok=True)
            logging.info(f"Directory successfully created at {path}")
        except Exception as e:
            logging.error(f"Erreur lors de la création du répertoire {path} : {e}")
            sys.exit(1)

def save_array_data(array_path, array):
    dir_path = os.path.dirname(array_path)
    os.makedirs(dir_path, exist_ok=True)  

    joblib.dump(array, array_path)
    print(f"Données sauvegardées dans {array_path}")


def save_obj(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'wb') as file_obj:
        joblib.dump(obj, file_obj)
    print(f"Objet sauvegardé dans {file_path}")

