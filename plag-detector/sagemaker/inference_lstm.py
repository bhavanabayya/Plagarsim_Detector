# sagemaker/inference_lstm.py
import os, json, joblib, tensorflow as tf
from pathlib import Path
from src.preprocess import clean_text
from src.utils_text import pair_to_arrays

MAX_LEN = 60

def model_fn(model_dir):
    model = tf.keras.models.load_model(Path(model_dir) / "lstm_model")
    tok = joblib.load(Path(model_dir) / "tokenizer.pkl")
    return {"model": model, "tok": tok}

def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError("Only application/json is supported")
    return json.loads(body)

def predict_fn(data, artifacts):
    model, tok = artifacts["model"], artifacts["tok"]
    src = clean_text(data.get("source_txt",""))
    sus = clean_text(data.get("plagiarism_txt",""))
    s_ids, p_ids = pair_to_arrays(src, sus, tok, MAX_LEN)
    proba = float(model.predict([s_ids, p_ids], verbose=0)[0][0])
    return {"plag_percent": round(proba*100, 2)}

def output_fn(pred, accept="application/json"):
    return json.dumps(pred), "application/json"
