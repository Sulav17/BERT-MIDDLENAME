from fastapi import FastAPI
from app.schema import FullNameRequest, MiddleNameResponse
from app.model import predict_middle_name

app = FastAPI(title="BERT-MIDDLENAME API")

@app.get("/")
def root():
    return {"message": "Welcome to BERT-MIDDLENAME API"}

@app.post("/predict-middle-name", response_model=MiddleNameResponse)
def get_middle_name(data: FullNameRequest):
    middle = predict_middle_name(data.full_name)
    return {
        "full_name": data.full_name,
        "middle_name": middle
    }

