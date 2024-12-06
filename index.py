from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import requests
import json

# Load the saved models
rfc_model = joblib.load('rfc_model.pkl')  # Random Forest Classifier
svc_model = joblib.load('svc_model.pkl')  # Support Vector Classifier

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Or specify allowed methods like ['POST']
    allow_headers=["*"],  # Or specify allowed headers
)

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

class InputData(BaseModel):
    rfid: str
    temperature: str
    
# Prediction route using Random Forest Classifier
@app.post("/predict_rfc")
async def predict_rfc(copd_data: COPDData):
    # Prepare features
    features = prepare_features(copd_data)
    # Make prediction using RFC model
    prediction = rfc_model.predict(features)
    print(prediction.tolist())
    return {"rfc_prediction": prediction.tolist()}

# Prediction route using Support Vector Classifier
@app.post("/predict_svc")
async def predict_svc(copd_data: COPDData):
    # Prepare features
    features = prepare_features(copd_data)
    
    # Make prediction using SVC model
    prediction = svc_model.predict(features)
    return {"svc_prediction": prediction.tolist()}


# Prediction route using Support Vector Classifier
@app.post("/send_data")
async def save_to_ipfs(input_data: InputData):            
    # IPFS API endpoint
    ipfs_api_url = "http://127.0.0.1:5001/api/v0/add"

    # JSON data to store
    data = {
        "name": "Example3",
        "description": "This is a sample JSON stored on IPFS. by IOT Actual",
        "timestamp": "2024-12-06T12:00:00Z"
    }

    # Convert JSON data to a file-like object
    files = {
        'file': ('dataByIotActual.json', json.dumps(data))
    }

    # Send a POST request to the IPFS API
    response = requests.post(ipfs_api_url, files=files)

    # Check response
    if response.status_code == 200:
        result = response.json()
        print(f"JSON data stored in IPFS with CID: {result['Hash']}")
        return {"Ipfs-Hash":result['Hash']}
    else:
        print(f"Failed to upload data: {response.text}")
        return {"Ipfs-Hash":"None"}



