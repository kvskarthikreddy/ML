from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from emotion_model import predict_emotion  # Importing emotion detection function

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input model
class TextInput(BaseModel):
    text: str

# API route for emotion detection
@app.post("/predict")
def predict(input_data: TextInput):
    result = predict_emotion(input_data.text)
    return {"emotions": result}

# Root endpoint for checking if API is running
@app.get("/")
def read_root():
    return {"message": "Emotion Detection API is running!"}
