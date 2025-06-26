import torch
from transformers import BertTokenizer, BertForSequenceClassification
from app.utils.label_map import id_to_middle_name
import os
from pathlib import Path
import torch.nn.functional as F

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

def predict_middle_name(full_name: str) -> list:
    parts = full_name.strip().split()

    if len(parts) == 2:
        return [{"name": "no middlename", "score": 1.0}]

    load_model()
    inputs = tokenizer(full_name, return_tensors="pt", truncation=True, padding=True, max_length=32)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0] 

    valid_predictions = []
    for class_idx, prob in enumerate(probs):
        score = prob.item()
        print(f"Class {class_idx} - Score: {score}")
        if score > 0.0:  
            class_name = id_to_middle_name.get(class_idx, f"unknown_{class_idx}")
            valid_predictions.append({
                "name": class_name,
                "score": round(score, 9)
            })

    print(f"🔍 Non-zero predictions: {valid_predictions}")

    valid_predictions.sort(key=lambda x: x["score"], reverse=True)

    if len(valid_predictions) >= 2 and valid_predictions[1]["score"] > 0.1:
        return valid_predictions[:2]
    elif len(valid_predictions) >= 1:
        return [valid_predictions[0]]
    else:
        return [{"name": "no middlename", "score": 1.0}]