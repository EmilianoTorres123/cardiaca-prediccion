from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

origins = ["*"]

app = FastAPI(title='Heart failure Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

model = load(pathlib.Path('model/heart_failure_clinical_records_dataset-v1.joblib'))

class InputData(BaseModel):
    age: int = 75
    anaemia: int = 1
    creatinine_phosphokinase: int = 582
    diabetes: int = 0
    ejection_fraction: int = 38
    high_blood_pressure: int = 1
    platelets: int = 265000
    serum_creatinine: float = 9.4
    serum_sodium: int = 140
    sex: int = 0
    smoking: int = 1
    time: int = 4

class OutputData(BaseModel):
    prediction: int
    probability: float 

@app.post('/score', response_model=OutputData)
def score(data: InputData):
    model_input = np.array([v for k, v in data.dict().items()]).reshape(1, -1)
    probability = model.predict_proba(model_input)[:, -1]
    
    # Realiza una predicción binaria (1 o 0) a partir de la probabilidad
    prediction = 1 if probability >= 0.5 else 0

    return {'prediction': prediction, 'probability': probability}
