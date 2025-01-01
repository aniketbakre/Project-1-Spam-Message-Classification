from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
# from fastapi.middleware.cors import CORSMiddleware
import joblib
import os

# Initializing FastAPI app
app = FastAPI()

# mounting css file
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Get the absolute path of the directory containing this file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load model
try:
    rf_model = joblib.load(os.path.join(MODELS_DIR, "model_Random_Forest_Classifier.pkl"))
    svm_model = joblib.load(os.path.join(MODELS_DIR, "model_svm.pkl"))
except FileNotFoundError as e:
    raise FileNotFoundError(f"Model file not found in the 'models' directory: {e}")

# Home route for HTML
@app.get("/", response_class = HTMLResponse)
async def serve_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Predict route
@app.post("/predict/")
async def predict_message(
    message: str = Form(...),
    model: str = Form(...)
):
    # To select the model
    if model == "model_random_Forest_Classifier.pkl":
        selected_model = rf_model
    elif model == "model_svm.pkl":
        selected_model = svm_model
    else:
        return JSONResponse(
            content={"error": "Invalid model selected. Please select a valid model."},
            status_code=400
        )

    

    # Perform prediction
    try:
        prediction = selected_model.predict([message])[0]
        result = "Alert!!! Its a 'Spam message', Please be cautious. Don't click on any links or reply." if prediction == "spam" else "Not Spam"
    except Exception as e:
        return JSONResponse(
            content={"error": f"Prediction failed: {str(e)}"},
            status_code=500
        )

    return {
        "message": message,
        "model_used": model,
        "prediction": result
    }
