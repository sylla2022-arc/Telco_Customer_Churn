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
import time
import os

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

@app.get("/")
def docs_page():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        logging.info(f"Data preview:\n{df.head(3)}")

        df = clean_input_data(df)
        if "Churn" in df.columns:
            logging.info("Cible détectée dans le fichier uploadé → monitoring de performance possible.")
        else:
            logging.info("Pas de cible → seulement monitoring de drift/distribution possible.")

        preprocessor = load_obj("artifact/data_transformation/preprocessor/preprocessor.pkl")
        model = load_obj("artifact/model_trainer/save_model/model.pkl")

        trans_df = preprocessor.transform(df)
        y_pred = model.predict(trans_df)
        df['Churn_Prediction'] = y_pred

        output_path = f"artifact/predict_data/predictions_{int(time.time())}.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        return {
            "status": "success",
            "preview": df.head(10).to_dict(orient="records"),
            "output_file": output_path,
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
        current_file = sorted([f for f in os.listdir("artifact/predict_data") if f.endswith('.csv')])[-1]
        current_data = pd.read_csv(f"artifact/predict_data/{current_file}")
        
        # Convertir Yes/No en 1/0
        reference_data['Churn_numeric'] = reference_data['Churn'].map({'Yes': 1, 'No': 0})
        current_data['Churn_numeric'] = current_data['Churn'].map({'Yes': 1, 'No': 0})

         # Taux de churn
        if "Churn_numeric" in current_data.columns:
            churn_rate = current_data["Churn_numeric"].value_counts(normalize=True).get(1, 0.0)
        else:
            churn_rate = 0.0
        
        # Configuration Evidently
        column_mapping = ColumnMapping()
        column_mapping.target = 'Churn_numeric'
        column_mapping.numerical_features = ['tenure', 'SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
        exclude_cols = set(column_mapping.numerical_features) | {column_mapping.target, 'Churn'} | {'customerID'}
        column_mapping.categorical_features = [col for col in reference_data.columns if col not in exclude_cols]
        
        # Target Drift
        target_drift_report = Report(metrics=[ DataDriftPreset(), TargetDriftPreset()])
        target_drift_report.run(
            reference_data=reference_data, 
            current_data=current_data,
            column_mapping=column_mapping
        )
        monitoring_dir = "artifact/monitoring/target_drift.html"
        os.makedirs(os.path.dirname(monitoring_dir), exist_ok=True)
    
        target_drift_report.save_html(monitoring_dir)

       
            
        # Résultats
        report_data = target_drift_report.as_dict()
        nb_clients = len(current_data)
        report_data['nb_clients'] = nb_clients
        
        drift_detected = False
        drift_score = 0.0
        
        for metric in report_data['metrics']:
    
            if 'result' in metric and 'dataset_drift' in metric['result']:
                drift_detected = metric['result']['dataset_drift']
                drift_score = metric['result'].get('drift_score', 0.0)
                
                break
        
        return {
            "target_drift_detected": drift_detected,
            "drift_score": round(drift_score, 3),
            "file": current_file,
            "nb_clients": nb_clients,
            "churn_rate": round(churn_rate, 3),
            "message": "No  drift detected in target distribution"
        }
        
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)






