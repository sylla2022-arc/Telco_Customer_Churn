import os
import yaml
from src.Churn_Project.logging.logger import logging
import sys
from box import ConfigBox #type:ignore
import joblib
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from catboost import CatBoostClassifier

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

def load_array_data(array_path):

    if not os.path.exists(array_path):
        logging.error(f"Le fichier yaml {array_path} n'existe pas.")
        raise FileNotFoundError(f"Le fichier yaml {array_path} n'existe pas.")

    with open(array_path, 'rb') as file:
        content = joblib.load(file)
        logging.info(f"Numpy array sucessfully loaded from {array_path}")
        print(f"Numpy array sucessfully loaded from {array_path}")
        return content


def save_obj(file_path, obj):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, 'wb') as file_obj:
        joblib.dump(obj, file_obj)
    print(f"Objet sauvegardé dans {file_path}")

def load_obj(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Le fichier {file_path} n'existe pas.")
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    with open(file_path, 'rb') as file_obj:
        obj = joblib.load(file_obj)
        logging.info(f"Objet chargé avec succès depuis {file_obj}")
        print(f"Objet chargé avec succès depuis {file_obj}")
        return obj

def load_obj(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Le fichier {file_path} n'existe pas.")
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")

    with open(file_path, 'rb') as file_obj:
        obj = joblib.load(file_obj)
        logging.info(f"Numpy array sucessfully loaded from {file_obj}")
        print(f"Numpy array sucessfully loaded from {file_obj}")
        return obj
    


def hyperparameter_tuning(X_train, y_train, model_name, param_grid,
                          use_random_search=False, n_iter=10  ):
    if model_name == "CatBoostClassifier":
        model = CatBoostClassifier(verbose=100, random_state=42, class_weights = [1, 2])

    else:
        raise ValueError(f"Unknown model name: {model_name}")
    if use_random_search and param_grid:
                grid = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=3,
                    n_jobs=-1,
                    refit=True,
                    verbose=1,
                    random_state=42
                )
    else:
        grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        cv=3,
                        refit=True,
                        n_jobs=-1,
                        verbose=1
    )
    grid.fit(X_train, y_train)
    return grid

