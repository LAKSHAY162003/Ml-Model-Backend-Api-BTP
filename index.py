from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the saved models
rfc_model = joblib.load('rfc_model.pkl')  # Random Forest Classifier
svc_model = joblib.load('svc_model.pkl')  # Support Vector Classifier

# Initialize FastAPI app
app = FastAPI()

# Define the input data model with the exact column names
class COPDData(BaseModel):
    ID: int
    AGE: float
    PackHistory: float
    COPDSEVERITY: float
    MWT1: float
    MWT2: float
    MWT1Best: float
    FEV1: float
    FEV1PRED: float
    FVC: float
    FVCPRED: float
    CAT: float
    HAD: float
    SGRQ: float
    AGEquartiles: int
    gender: int
    smoking: int
    Diabetes: int
    muscular: int
    hypertension: int
    AtrialFib: int
    IHD: int

# Function to convert the COPDData model to the format required by the model
def prepare_features(data: COPDData):
    # Exclude ID and copd (target) fields
    feature_list = [
        data.AGE, 
        data.PackHistory, 
        data.MWT1Best, 
        data.FEV1, 
        data.FEV1PRED, 
        data.FVC, 
        data.FVCPRED, 
        data.CAT, 
        data.HAD, 
        data.SGRQ, 
        data.AGEquartiles, 
        data.gender, 
        data.smoking, 
        data.Diabetes, 
        data.muscular, 
        data.hypertension, 
        data.AtrialFib, 
        data.IHD
    ]
    return np.array(feature_list).reshape(1, -1)

# Prediction route using Random Forest Classifier
@app.post("/predict_rfc")
async def predict_rfc(copd_data: COPDData):
    # Prepare features
    print("success")
    features = prepare_features(copd_data)
    print("success")
    # Make prediction using RFC model
    prediction = rfc_model.predict(features)
    return {"rfc_prediction": prediction.tolist()}

# Prediction route using Support Vector Classifier
@app.post("/predict_svc")
async def predict_svc(copd_data: COPDData):
    # Prepare features
    features = prepare_features(copd_data)
    
    # Make prediction using SVC model
    prediction = svc_model.predict(features)
    return {"svc_prediction": prediction.tolist()}
