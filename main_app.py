from fastapi import FastAPI
from pydantic import BaseModel
from predicting import Predictor
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
predictor = Predictor()

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn gốc (thay đổi theo yêu cầu của bạn)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức HTTP
    allow_headers=["*"],  # Cho phép tất cả các header
)

class Data(BaseModel):
    model: str
    title: str
    content: str

@app.get("/")
def read_root():
    return {"<h1>Jv. WELCOME TO MY SERVER</h1>"}

@app.post("/analyze-sentiment/")
def analyze_sentiment(data: Data):
    prediction = []
    result = predictor.predict_text(data.model, {'title': data.title, 'content': data.content})
    prediction.append(int(result[0]))
    return {"prediction": prediction}