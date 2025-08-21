from src.Churn_Project.utils.utility import load_array_data, load_obj, hyperparameter_tuning
from src.Churn_Project.entity.config_entity import DataTransformationConfig
from src.Churn_Project.constants import PARAM_YAML_FILE_PATH
from src.Churn_Project.utils.utility import read_yaml_file
from src.Churn_Project.logging.logger import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from urllib.parse import urlparse
from mlflow.models import infer_signature
import shutil

import mlflow
import os

class ModelTrainer :
    def __init__(self, config, data_transformation_config :DataTransformationConfig):
        self.config = config
        self.data_transformation_config = data_transformation_config
    
    def mlflow_tracking(self, train_accuracy, test_accuracy, best_model, best_params,
                        classification_report, confusion_matrix, signature):
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment("ModelTrainer")

        with mlflow.start_run(run_name=f"Test {best_model}") :
            mlflow.log_metrics({'Accuracy train ': train_accuracy,
                                'Accuracy test ': test_accuracy,
                               })


            mlflow.log_params({'best_n_estimators': best_params['n_estimators'],
                                'best_max_depth': best_params['max_depth'],
                               'best_learning_rate': best_params['learning_rate']
            })

            mlflow.log_text(str(classification_report), "classification_report.txt")
            mlflow.log_text(str(confusion_matrix), "confusion_matrix.txt")

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != 'file' :
                mlflow.sklearn.log_model(best_model, artifact_path="best_model", 
                                         registered_model_name="CatBoostClassifier")

            else :
                mlflow.sklearn.log_model(best_model, artifact_path="best_model", signature=signature)

            model_path = "artifact/model_trainer/save_model"

            model_path = "artifact/model_trainer/save_model"
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
            os.makedirs(model_path, exist_ok=True)

            mlflow.sklearn.save_model(best_model, model_path)
            mlflow.log_artifacts(model_path, artifact_path="model")
            shutil.rmtree(model_path)

            print(f"Modèle {best_model} sauvegardé - accuracy_train: {train_accuracy:.4f}, test_accuracy: {test_accuracy:.4f}")

    def initiate_model_trainer(self):
        try:
            # charge the data
            logging.info('Model initiation start ...')
            X_train =  load_array_data(self.data_transformation_config.train_file_path)
            X_test =  load_array_data(self.data_transformation_config.test_file_path)
            y_train = load_array_data(self.data_transformation_config.y_train_array)
            y_test = load_array_data(self.data_transformation_config.y_test_array)

            # Load preprocessor
            #preprocessor = load_obj(file_path=self.data_transformation_config.preprocessor_obj)
            
            # Load parms grid
            param_grid_cat = read_yaml_file(path_yml=PARAM_YAML_FILE_PATH)

            grid_search_cv = hyperparameter_tuning(X_train=X_train, y_train=y_train,
                                                   model_name=param_grid_cat["model_name"], 
                                                   param_grid=param_grid_cat['model_params'],
                                                   use_random_search=True, n_iter=10 )
            
            # get the bestparams
            best_params = grid_search_cv.best_params_
            best_model = grid_search_cv.best_estimator_

            logging.info(f'Best params: {best_params}')
            logging.info(f'Best estimator: {best_model}')
            print(f'Best params: {best_params}')
            print(f'Best estimator: {best_model}')

            #Prediction
            y_pred_train = best_model.predict(X_train)
            y_pred_test = best_model.predict(X_test)

            y_pred_proba = best_model.predict_proba(X_test)[:,1]

            # for thresh in [0.3, 0.4, 0.5]:
            #     y_pred = (y_pred_proba > thresh).astype(int)
            #     print(f"\nThreshold={thresh}")
            #     print(classification_report(y_test, y_pred))

            train_accuracy = accuracy_score(y_true=y_train, y_pred=y_pred_train)
            test_accuracy = accuracy_score(y_true=y_test, y_pred=y_pred_test)

            logging.info(f'Accuracy score train set: {train_accuracy}')
            logging.info(f'Accuracy score test set: {test_accuracy}')
            print(f'Accuracy score train set: {train_accuracy}')
            print(f'Accuracy score test set: {test_accuracy}')

            # get classifocation
            cm = confusion_matrix(y_true=y_test, y_pred=y_pred_test)
            cr = classification_report(y_true=y_test, y_pred=y_pred_test)
            signature= infer_signature(X_train, y_train)

            self.mlflow_tracking(train_accuracy =train_accuracy, test_accuracy = test_accuracy,
                                  best_model = best_model, best_params =best_params ,
                        classification_report = cr, confusion_matrix = cm, signature = signature)
        except Exception as e:
            logging.error(f'Error : {e}')
            raise e