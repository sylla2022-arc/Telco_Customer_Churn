import pandas as pd
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, UploadFile, File
import uvicorn 
from starlette.responses import RedirectResponse
from src.Churn_Project.logging.logger import logging
from src.Churn_Project.utils.utility import load_obj
from main import Main
import time
import os


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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)






