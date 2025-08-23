import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.Churn_Project.utils.utility import save_array_data, save_obj, create_directory
from src.Churn_Project.entity.config_entity import DataTransformationConfig
from src.Churn_Project.logging.logger import logging


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def _features_enginneering(self, df: pd.DataFrame) -> pd.DataFrame:
        # Validation des colonnes nécessaires
        #print(df.head(2))
        required_columns = ['TotalCharges', 'MonthlyCharges', 'Churn']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Les colonnes suivantes sont manquantes : {missing_columns}")

        # Nettoyage et transformation
        df = df.copy()
        df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
        df['TotalCharges'] = df['TotalCharges'].replace('', np.nan)
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        if "customerID" in df.columns:
            df.drop(columns='customerID', axis = 1,  inplace=True)
        else :
            print(f"La colonne customerID n'existe pas")
            logging.info(f"La colonne customerID n'existe pas")
        
        return df

    def get_preprocessor_object(self, numerical_features, cat_feature):
        try:
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])
            
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))  
            ])
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, cat_feature)
            ])
            return preprocessor
        except Exception as e:
            logging.error(f"Erreur lors de la création du préprocesseur : {e}")
            raise e

    def initiate_data_transformation(self):
        try:
            # Vérification  du fichier
            if not os.path.exists(self.config.data_dir):
                logging.error(f"Fichier introuvable : {self.config.data_dir}")
                raise FileNotFoundError(f"Fichier introuvable : {self.config.data_dir}")

            # Chargement
            df = pd.read_csv(self.config.data_dir)
            logging.info(f"Fichier chargé avec succès depuis {self.config.data_dir}")

            # Ingenierie des features
            df = self._features_enginneering(df)

          
            X = df.drop(columns='Churn')
            y = df['Churn']


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logging.info(f"Données divisées en ensembles d'entraînement et de test")

   
            directories = [
                os.path.dirname(self.config.train_file_path),
                os.path.dirname(self.config.test_file_path),
                os.path.dirname(self.config.y_train_array),
                os.path.dirname(self.config.y_test_array)
            ]
            create_directory(directories)
            logging.info("Répertoires créés avec succès.")

        
            num_cols = X.select_dtypes(exclude='object').columns.tolist()
            cat_cols = X.select_dtypes(include='object').columns.tolist()

            # Création et application du préprocesseur
            preprocessor_obj = self.get_preprocessor_object(numerical_features=num_cols, cat_feature=cat_cols)
            preprocessor_obj.fit(X_train)
            train_transformed = preprocessor_obj.transform(X_train)
            test_transformed = preprocessor_obj.transform(X_test)

            # Sauvegarde des données transformées et du préprocesseur
            save_array_data(array_path=self.config.y_train_array, array=np.array(y_train))
            save_array_data(array_path=self.config.y_test_array, array=np.array(y_test))
            save_array_data(array_path=self.config.train_file_path, array=train_transformed)
            save_array_data(array_path=self.config.test_file_path, array=test_transformed)
            
            save_obj(file_path=self.config.preprocessor_obj, obj=preprocessor_obj)
            logging.info("Données transformées et préprocesseur sauvegardés avec succès.")

        except Exception as e:
            logging.error(f"Erreur lors de la transformation des données : {e}")
            raise e