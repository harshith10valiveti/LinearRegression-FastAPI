from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initializing the FastAPI application instance
app = FastAPI()

# Loading the trained regression model using joblib
model = joblib.load("boston_model.pkl")

# Defining a pydantic base model 
class HouseData(BaseModel):
    RM: float
    LSTAT: float
    PTRATIO: float

# Home route
@app.get("/")
def home():
    return {"message": "Welcome to the Boston Housing Price Prediction API"}

# Prediction route
@app.post("/predict/")
def predict_price(data: HouseData):           #house datastructure
    # Prepare data for prediction, collecting the input data and converting into an array for model input
    features = np.array([[data.RM, data.LSTAT, data.PTRATIO]])
    
    # Predicting house price
    predicted_price = model.predict(features)[0]

    return {
        "Predicted_Price": round(predicted_price * 1000, 2),  # Converting back to USD
    }

@app.put("/update/", tags=["update"])
def update_data(data: HouseData):
    # Example logic: Update parameters or features
    updated_features = {
        "RM": data.RM + 0.1,
        "LSTAT": data.LSTAT - 0.1,
        "PTRATIO": data.PTRATIO
    }

    return {"message": "Data updated successfully", "updated_features": updated_features}

@app.delete("/delete/")
def delete_data():
    return {"message": "Prediction data deleted successfully"}
