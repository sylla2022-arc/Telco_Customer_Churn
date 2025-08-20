import os
import sys
import pandas as pd
import numpy as np
from src.Churn_Project.entity.config_entity import DataValidationConfig
from src.Churn_Project.logging.logger import logging
from src.Churn_Project.utils.utility import  create_directory
import mlflow


root_path = os.path.abspath(os.path.join(os.getcwd(), "../."))
if root_path not in sys.path:
    sys.path.append(root_path)
print(root_path)

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, DataQualityPreset


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('Data Validation')
    
    def _clean_total_charges(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
        df['TotalCharges'] = df['TotalCharges'].replace('', np.nan)
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        return df
    
    def _generate_evidently_report(self, df, reference_df=None) -> Report:
        """Génère un rapport Evidently"""
    
        reference_path = "data_churn/reference.csv"

        # Première fois : on sauvegarde df commee réf
        if not os.path.exists(reference_path):

            df = self._clean_total_charges(df)

            # Sauvegarder la référence propre
            df.to_csv(reference_path, index=False, header=True)
            reference_df = df.copy()
            
        else:
            reference_df = pd.read_csv(reference_path)

        df = self._clean_total_charges(df) 
    
        # Définition des features
        df_col_cat = df.select_dtypes(include=['object']).columns.to_list()
        df_col_num = df.select_dtypes(exclude=['object']).columns.to_list()
        #df_col_num = ['tenure', "MonthlyCharges"]

        data_def = ColumnMapping(
            categorical_features=df_col_cat,
            numerical_features=df_col_num,
            target=list(self.config.target.keys())[0] if self.config.target else None
        )

        # Création du rapport
        report = Report(metrics=[DataQualityPreset(),
                                DataDriftPreset()])

        report.run(
            current_data=df, 
            reference_data=reference_df,
            column_mapping=data_def
        )
        
        return report


    def initiate_data_validation(self):
        logging.info('Data validation initiated')
        with mlflow.start_run():
            try:
                if not os.path.exists(self.config.data_dir):
                    raise FileNotFoundError(f"Le fichier source est introuvable : {self.config.data_dir}")

                df = pd.read_csv(self.config.data_dir)
                logging.info(f"Data loaded successfully from {self.config.data_dir}")

                # Log basid data info to mlflow
                mlflow.log_param('data_path', self.config.data_dir)
                mlflow.log_param("num_sample", len(df))
                mlflow.log_param('num_features', df.shape[1])
                logging.info(f"log params  to mlflow commpleted")

                print(f"The number of columns is {df.shape[1]}")
                # cols reelles
                actual_cols = set(df.columns)
                logging.info(f'Type of all_feature: {type(self.config.feature)}')
                logging.info(f'Values of features {self.config.feature}')
                logging.info(f"Target column value: {self.config.target}")

                expected_features = set(self.config.feature.keys())
                expected_target = set(self.config.target.keys())

                print(f'expected features: {expected_features}')
                print(f'expected target: {expected_target}')
            
                # Colonnes manquantes 
                missing_features = expected_features - actual_cols
                missing_target = expected_target - actual_cols

                print(f'Missing features: {missing_features}')
                print(f'Missing target: {missing_target}')

                # Colonnes en trop 
                allowed_columns = expected_features | expected_target
                extra_cols = actual_cols - allowed_columns

                # validaion
                status_feature = len(missing_features) == 0 and len(extra_cols) == 0
                statut_target = len(missing_target) == 0

                # Suivi
                if missing_features:
                    logging.warning(f"Missing columns: {missing_features}")
                    print(f"Missing columns: {missing_features}")

                if extra_cols:
                    logging.warning(f"Unexpected columns: {extra_cols}")
                    print(f"Unexpected columns: {extra_cols}")
                
                if missing_target:
                    logging.warning(f"Missing target: {missing_target}")
                    print(f"Missing target: {missing_target}")

                  # Log mettric to mlflow
                mlflow.log_metric('missing_features_count', len(missing_features))
                mlflow.log_metric('missing_target_count', len(missing_target))
                mlflow.log_metric('extra_cols_count', len(extra_cols))
                mlflow.log_metric('Validation statu feature', status_feature)
                mlflow.log_metric('Validation statu target', statut_target)
                logging.info(f"log metrics to mlflow commpleted")

                # generation du rapport evidently
                evidently_report = self._generate_evidently_report(df, reference_df=None)
            

                evidently_report.save_html('evidently_report.html')
                mlflow.log_artifact('evidently_report.html')

                report_dict = evidently_report.as_dict()

                # Exemple: loguer les résultats principaux du DataDriftPreset
                for metric in report_dict['metrics']:
                    if metric['metric'] == 'DataDriftPreset':
                        drift_result = metric.get('result', {})
                        mlflow.log_metric("share_drifted_columns", drift_result.get("share_drifted_columns", 0))
                        mlflow.log_metric("number_of_drifted_columns", drift_result.get("number_of_drifted_columns", 0))
                        mlflow.log_metric("drift_detected", int(drift_result.get("drift_detected", False)))


                with open(self.config.status_file, "w") as f:
                    f.write(f"Validation status (features): {status_feature}\n")
                    f.write(f"Missing features: {list(missing_features)}\n")
                    f.write(f"Validation status (target): {statut_target}\n")
                    f.write(f"Missing target: {list(missing_target)}\n")
                    f.write(f"Unexpected columns: {list(extra_cols)}\n")

                logging.info(f"Data validation commpleted")
                return status_feature, statut_target

            except Exception as e:
                logging.error(f"Une erreur s'est glissée: {e}")
                mlflow.log_param("Erreur", str(e))
                raise e
