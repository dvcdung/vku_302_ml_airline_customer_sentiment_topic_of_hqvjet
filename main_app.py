from fastapi import FastAPI
from pydantic import BaseModel
from predicting import Predictor


app = FastAPI()
predictor = Predictor()

class Data(BaseModel):
    model: str
    title: str
    content: str

@app.get("/")
def read_root():
    return {"<h1>Jv. WELCOME TO MY SERVER</h1>"}

@app.post("/analyze-sentiment/")
def analyze_sentiment(data: Data):
    result = predictor.predict_text(data.model, {'title': data.title, 'content': data.content})
    return {"result": "Positive" if result[0] == 2 else "Neutral" if result[0] == 1 else "Negative"}