from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

# Create the FastAPI instance
app = FastAPI()



@app.post("/faurd-detection")
async def faurd_detection(request: Request):
    body = await request.json()
    step = body["step"]
    amount = body["amount"]
    oldbalanceOrg = body["oldbalanceOrg"]
    newbalanceOrig = body["newbalanceOrig"]
    oldbalanceDest = body["oldbalanceDest"]
    newbalanceDest = body["newbalanceDest"]
    
    
    input_data = pd.DataFrame(
        {"step": step, "amount": amount, "oldbalanceOrg": oldbalanceOrg, "newbalanceOrig": newbalanceOrig, "oldbalanceDest": oldbalanceDest, "newbalanceDest": newbalanceDest}, index=[0]
    )
    model_path = "Pipeline_XGB_model.joblib"
    input_model = joblib.load(model_path)
    input_data = input_model.transform(input_data)

    model_path = "XGB_model.joblib"
    model = joblib.load(model_path)
    prediction = model.predict(input_data)
    print("Prediction", prediction)
    response_data = {"prediction": str(prediction)}
    return JSONResponse(response_data)

  
