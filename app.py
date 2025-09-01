import pandas as pd
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, UploadFile, File
import uvicorn 
from starlette.responses import RedirectResponse
from src.Churn_Project.logging.logger import logging
from src.Churn_Project.utils.utility import load_obj

from main import Main
from src.Churn_Project.logging.logger import logging

import time
import os
import io
from typing import List

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
from evidently import ColumnMapping
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

templates = Jinja2Templates(directory="./templates")

app = FastAPI()
origin = ['*']

app.add_middleware(CORSMiddleware,
    allow_origins=origin,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_input_data(df):
    """Identique à celui employé dans data transfo."""
    df = df.copy()
    df['TotalCharges'] = df['TotalCharges'].astype(str).str.strip()
    df['TotalCharges'] = df['TotalCharges'].replace('', np.nan)
    df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
    df['TotalCharges'] = df['TotalCharges'].astype(float)

    if "customerID" in df.columns:
        df.drop(columns='customerID', inplace=True)
    else:
        logging.info("La colonne customerID n'existe pas dans ce fichier uploadé")
    
    return df

def choice_valid_file_global( pred_files : List) ->  tuple:
    valid_files = []
    for file in pred_files:
        df_temp = pd.read_csv(f"artifact/predict_data/{file}")
        if len(df_temp) > 100:  # Seulement les fichiers avec + de 100 lignes
            valid_files.append((file, len(df_temp)))
    
    
    if not valid_files:
        return {"error": "Aucun fichier global trouvé (seulement des échantillons)"}

    # Prendre le plus gros fichier
    current_file = max(valid_files, key=lambda x: x[1])[0]
    current_data = pd.read_csv(f"artifact/predict_data/{current_file}")
    logging.info(f"Fichier sélectionné: {current_file}, Taille: {len(current_data)} lignes")

    return current_data, current_file

def generate_drift_report(reference_data, current_data, column_mapping):
    """creer un rapport de dérive entre les données de réf et courantes"""
    target_drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    target_drift_report.run(
        reference_data=reference_data, 
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    return target_drift_report

def get_drift_metrics(report, current_data=None):
    """Extrait les metriques de dérive avec cacul des features impportances"""
    report_data = report.as_dict()
    
    # Init
    drift_detected = False
    drift_score = 0.0
    churn_distribution = {"Oui": 0, "Non": 0}
    
    # Extraire la dérive
    for metric in report_data['metrics']:
        if metric['metric'] == 'DatasetDriftMetric':
            if 'result' in metric and 'dataset_drift' in metric['result']:
                drift_detected = metric['result']['dataset_drift']
                drift_score = metric['result'].get('drift_score', 0.0)
                break
    
    # Calc de la distribution
    if current_data is not None and 'Churn_numeric' in current_data.columns:
        churn_counts = current_data['Churn_numeric'].value_counts()
        logging.info(f"Les valeurs manuelles calculé des churn sont: {churn_counts}")
        churn_distribution = {
            "Oui": int(churn_counts.get(1, 0)),
            "Non": int(churn_counts.get(0, 0))
        }
    
    return drift_detected, drift_score, churn_distribution
@app.get("/")
def docs_page():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        logging.info(f"Data preview:\n{df.head(3)}")
        logging.info(f"Taille du dataset: {len(df)} lignes")

        df = clean_input_data(df)
        
        # Déterminer si c'est un dataset global ou un échantillon
        is_global = len(df) > 100  
        file_type = "global" if is_global else "sample" # Referer à la logique dans app sreamlit
        

        preprocessor = load_obj("artifact/data_transformation/preprocessor/preprocessor.pkl")
        model = load_obj("artifact/model_trainer/save_model/model.pkl")

        trans_df = preprocessor.transform(df)
        y_pred = model.predict(trans_df)
        df['Churn_Prediction'] = y_pred

        # Sauvegarder avec le type global ou sample
        output_path = f"artifact/predict_data/{file_type}_predictions_{int(time.time())}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, header= True)

        logging.info(f"Fichier sauvegardé: {output_path}, Type: {file_type}")

        return {
            "status": "success",
            "is_global": is_global,
            "file_type": file_type,
            "preview": df.head(10).to_dict(orient="records"),
   
            "dataset_size": len(df),
            "message": "Prédiction effectuée avec succès"
        }
    except Exception as e:
        logging.error(f"Erreur lors de la prédiction: {e}")
        raise e

@app.get("/train")
async def trainroute():
    train_pipeline = Main()
    train_pipeline.run_pipeline()
    return {"status": "Training pipeline completed successfully"}

@app.get("/target-drift")
async def target_drift():
    """Monitoring de la dérive de la target - Version finale"""
    try:
        # Charger les données
        reference_data = pd.read_csv("artifact/data_ingestion/raw/data_brute.csv")
        
        # Vérifier s'il y a des fichiers de prédiction
        pred_files = [f for f in os.listdir("artifact/predict_data") if f.endswith('.csv')]
        
        try:
            current_data, current_file = choice_valid_file_global(pred_files)
        except ValueError as e:
            return {"error": str(e)}

        # Convertir Yes/No en 1/0
        reference_data['Churn_numeric'] = reference_data['Churn'].map({'Yes': 1, 'No': 0})
        current_data['Churn_numeric'] = current_data['Churn'].map({'Yes': 1, 'No': 0})

        # Taux de churn
        if "Churn_numeric" in current_data.columns:
            churn_rate = current_data["Churn_numeric"].value_counts(normalize=True).get(1,0)
        else:
            churn_rate = 0.0
        
        # Configuration Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = 'Churn_numeric'
        column_mapping.numerical_features = ['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
        exclude_cols = set(column_mapping.numerical_features) | {column_mapping.target, 'Churn'} | {'customerID'}
        column_mapping.categorical_features = [col for col in reference_data.columns if col not in exclude_cols]
        
        # Générer le rapport
        target_drift_report = generate_drift_report(reference_data, current_data, column_mapping)
        
        # Sauvegarder le rapport HTML
        monitoring_dir = "artifact/monitoring/target_drift.html"
        os.makedirs(os.path.dirname(monitoring_dir), exist_ok=True)
        target_drift_report.save_html(monitoring_dir)
            
        # Extraire les métriques
        drift_detected, drift_score, churn_distribution = get_drift_metrics(target_drift_report, current_data=current_data)

        #recup les VRAIES données des features importantes
        important_features = {}
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        df_current_clean = clean_input_data(current_data)
        reference_data_clean = clean_input_data(reference_data)
        
        for feature in numerical_features:
            if feature in reference_data_clean.columns and feature in df_current_clean.columns:
                important_features[feature] = {
                    "reference": reference_data_clean[feature].tolist()[:20], 
                    "current": df_current_clean[feature].tolist()[:20]
                }
        

        return {
            "target_drift_detected": drift_detected,
            "drift_score": round(drift_score, 3),
            "file": current_file,
            "nb_clients": len(current_data),
            "churn_rate": round(churn_rate, 3),
            "churn_distribution": churn_distribution,
            "important_features": important_features,
            "message": "Drift analysis completed successfully"
        }
        
    except Exception as e:
        logging.error(f"Erreur dans target_drift: {str(e)}")
        return {"error": str(e)}

@app.post("/predict-sample-drift")
async def predict_sample_drift(file: UploadFile = File(...)):
    """Monitoring de dérive sur un échantillon personnalisé par rapport à la référence"""
    try:
        # Lire l'échantillon uploadé
        contents = await file.read()
        df_sample = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Réduire à 50 clients max pour l'analyse
        if len(df_sample) > 40:
            df_sample = df_sample.sample(40, random_state=42)
            logging.info(f"Échantillon réduit à 50 clients pour l'analyse")
        
        # Nettoyer les données de l'échantillon
        df_sample_clean = clean_input_data(df_sample)
        
        # Charger et nettoyer les données de RÉFÉRENCE
        reference_data = pd.read_csv("artifact/data_ingestion/raw/data_brute.csv")
        reference_data_clean = clean_input_data(reference_data) 
        
        # Convertir Yes/No en 1/0 pour les deux datasets
        reference_data_clean['Churn_numeric'] = reference_data_clean['Churn'].map({'Yes': 1, 'No': 0})
        
        if "Churn" in df_sample_clean.columns:
            df_sample_clean['Churn_numeric'] = df_sample_clean['Churn'].map({'Yes': 1, 'No': 0})
            churn_rate = df_sample_clean['Churn_numeric'].value_counts(normalize=True).get(1,0)
        else:
            churn_rate = 0.0
        
        # Configuration Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = 'Churn_numeric'
        column_mapping.numerical_features = ['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
        exclude_cols = set(column_mapping.numerical_features) | {column_mapping.target, 'Churn'} | {'customerID'}
        column_mapping.categorical_features = [col for col in reference_data_clean.columns if col not in exclude_cols]
        
        # Générer le rapport de dérive
        target_drift_report = generate_drift_report(reference_data_clean, df_sample_clean, column_mapping)
        
        # Extraire les métriques RÉELLES
        drift_detected, drift_score, churn_distribution = get_drift_metrics(target_drift_report, df_sample_clean)
        
        # recup les VRAIES données des features importantes
        important_features = {}
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        for feature in numerical_features:
            if feature in reference_data_clean.columns and feature in df_sample_clean.columns:
                important_features[feature] = {
                    "reference": reference_data_clean[feature].tolist()[:20], 
                    "current": df_sample_clean[feature].tolist()[:20]
                }
        
        # Statistiques descriptives
        sample_stats = {}
        if 'tenure' in df_sample_clean.columns:
            sample_stats['avg_tenure'] = round(df_sample_clean['tenure'].mean(), 2)
        if 'MonthlyCharges' in df_sample_clean.columns:
            sample_stats['avg_monthly_charges'] = round(df_sample_clean['MonthlyCharges'].mean(), 2)
        if 'TotalCharges' in df_sample_clean.columns:
            sample_stats['avg_total_charges'] = round(df_sample_clean['TotalCharges'].mean(), 2)
        
        return {
            "nb_clients": len(df_sample_clean),
            "churn_rate": round(churn_rate, 3),
            "drift_detected": drift_detected,
            "drift_score": round(drift_score, 3),
            "churn_distribution": churn_distribution,
            "important_features": important_features,
            "sample_stats": sample_stats,
            "message": "Sample drift analysis completed successfully"
        }
        
    except Exception as e:
        logging.error(f"Erreur dans predict_sample_drift: {str(e)}")
        return {"error": str(e)}

@app.get("/performance-metrics")
async def performance_metrics():
    """Retourne les métriques de performance du modèle"""
    try:
        # Charger les dernières prédictions
        pred_files = [f for f in os.listdir("artifact/predict_data") if f.endswith('.csv')]

        if not pred_files:
            return {"error": "Aucun fichier de prédiction trouvé"}
        
        current_data, current_file = choice_valid_file_global(pred_files=pred_files)

        # Vérifier si la cible est disponible
        if "Churn" not in current_data.columns or "Churn_Prediction" not in current_data.columns:
            return {"error": "Données insuffisantes pour calculer les métriques de performance"}
        
        # Convertir les labels
        y_true = current_data['Churn'].map({'Yes': 1, 'No': 0})
        y_pred = current_data['Churn_Prediction'].map({'Yes': 1, 'No': 0})
        
        # Calculer les métriques
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        return {
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
            'current_file' : current_file,
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }
        
    except Exception as e:
        logging.error(f"Erreur dans performance_metrics: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)