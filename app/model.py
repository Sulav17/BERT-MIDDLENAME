import torch
from transformers import BertTokenizer, BertForSequenceClassification
from app.utils.label_map import id_to_middle_name
import os
from pathlib import Path

_project_root = Path(__file__).parent.parent.resolve()
_default_model_dir = _project_root / "models" / "bert-middle-name"
model_path = os.getenv("MODEL_PATH", str(_default_model_dir))

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        print("🔄 Loading model and tokenizer from:", model_path)
        tokenizer = BertTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
        model.eval()

def predict_middle_name(full_name: str) -> str:
    load_model()
    inputs = tokenizer(full_name, return_tensors="pt", truncation=True, padding=True, max_length=32)
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()
    return id_to_middle_name[pred_id]
