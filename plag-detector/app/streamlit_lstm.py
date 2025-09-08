# # app/streamlit_lstm.py  (LOCAL-ONLY)
# import os, json
# import streamlit as st
# import pandas as pd
# from pathlib import Path
# from io import BytesIO

# # --- add '<repo-root>/src' to import path ---
# import sys
# ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(ROOT / "src"))

# from preprocess import clean_text
# from utils_text import pair_to_arrays
# from pypdf import PdfReader

# MODEL_PATH = Path("output/lstm_model.keras")   # local model (Keras 3)
# TOK_PATH   = Path("output/tokenizer.pkl")
# CORPUS_DIR = Path("corpus")
# MAX_LEN = 60

# st.set_page_config(page_title="LSTM Plagiarism Checker", page_icon="🧠", layout="wide")
# st.title("🧠 LSTM-based Plagiarism Checker (Local)")

# def read_text_file(uploaded):
#     return uploaded.read().decode("utf-8", errors="ignore")

# def read_pdf_file(uploaded):
#     reader = PdfReader(BytesIO(uploaded.read()))
#     return "\n".join([(p.extract_text() or "") for p in reader.pages])

# def chunk_text(t, max_words=120):
#     w = t.split()
#     return [" ".join(w[i:i+max_words]) for i in range(0, len(w), max_words)]

# @st.cache_resource
# def load_local_artifacts():
#     import keras          # Keras 3 API
#     import joblib
#     # Allow loading models that contain Lambda with python lambda
#     keras.config.enable_unsafe_deserialization()
#     model = keras.saving.load_model(
#         str(MODEL_PATH),
#         compile=False,
#         safe_mode=False
#     )
#     tok = joblib.load(TOK_PATH)
#     return model, tok

# def predict_local(source_txt: str, plag_txt: str) -> float:
#     model, tok = load_local_artifacts()
#     s_clean = clean_text(source_txt)
#     p_clean = clean_text(plag_txt)
#     s_ids, p_ids = pair_to_arrays(s_clean, p_clean, tok, MAX_LEN)
#     proba = float(model.predict([s_ids, p_ids], verbose=0)[0][0])
#     return round(proba * 100, 2)

# sus = st.text_area("Suspected text", height=200)
# src = st.text_area("Single source text (optional)", height=200)
# uploads = st.file_uploader("Or upload .txt/.pdf (multiple allowed)", type=["txt","pdf"], accept_multiple_files=True)

# if st.button("Check"):
#     if not sus.strip():
#         st.warning("Paste suspected text.")
#         st.stop()

#     candidates = []
#     if src.strip():
#         candidates.append(("textarea_source", src))
#     for f in uploads or []:
#         txt = read_text_file(f) if f.type == "text/plain" else read_pdf_file(f)
#         if txt.strip():
#             candidates.append((f.name, txt))
#     if CORPUS_DIR.exists():
#         for p in CORPUS_DIR.glob("**/*"):
#             if p.suffix.lower() not in {".txt", ".pdf"}:
#                 continue
#             try:
#                 text = p.read_text(encoding="utf-8", errors="ignore") if p.suffix==".txt" else ""
#                 if text.strip():
#                     candidates.append((str(p), text))
#             except Exception:
#                 pass
#     if not candidates:
#         candidates = [("self", sus)]

#     rows = []
#     for name, raw in candidates:
#         for chunk in chunk_text(raw):
#             pct = predict_local(chunk, sus)
#             rows.append({
#                 "source": name,
#                 "chunk_preview": chunk[:160].replace("\n"," ") + ("…" if len(chunk) > 160 else ""),
#                 "plag_percent": pct
#             })

#     df = pd.DataFrame(rows).sort_values("plag_percent", ascending=False)
#     st.subheader("Top matches")
#     st.dataframe(df.head(20), use_container_width=True)
#     if not df.empty:
#         st.metric("Highest plagiarism likelihood", f"{df.iloc[0]['plag_percent']:.2f} %")

# st.caption("Mode: **Local model** · File: output/lstm_model.keras")
# app/streamlit_lstm.py  (LOCAL-ONLY)
import os, json
import streamlit as st
import pandas as pd
from pathlib import Path
from io import BytesIO

# --- add '<repo-root>/src' to import path ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocess import clean_text
from utils_text import pair_to_arrays
from pypdf import PdfReader

# Fix for Lambda layer deserialization error
import tensorflow as tf
tf.keras.utils.get_custom_objects().clear()  # Clear any existing custom objects
import keras
keras.config.enable_unsafe_deserialization()

MODEL_PATH = Path("output/lstm_model.keras")   # local model (Keras 3)
TOK_PATH   = Path("output/tokenizer.pkl")
CORPUS_DIR = Path("corpus")
MAX_LEN = 60

st.set_page_config(page_title="LSTM Plagiarism Checker", page_icon="🧠", layout="wide")
st.title("🧠 LSTM-based Plagiarism Checker (Local)")

def read_text_file(uploaded):
    return uploaded.read().decode("utf-8", errors="ignore")

def read_pdf_file(uploaded):
    reader = PdfReader(BytesIO(uploaded.read()))
    return "\n".join([(p.extract_text() or "") for p in reader.pages])

def chunk_text(t, max_words=120):
    w = t.split()
    return [" ".join(w[i:i+max_words]) for i in range(0, len(w), max_words)]

@st.cache_resource
def load_local_artifacts():
    import joblib
    model = keras.saving.load_model(
        str(MODEL_PATH),
        compile=False,
        safe_mode=False
    )
    tok = joblib.load(TOK_PATH)
    return model, tok

def predict_local(source_txt: str, plag_txt: str) -> float:
    model, tok = load_local_artifacts()
    s_clean = clean_text(source_txt)
    p_clean = clean_text(plag_txt)
    s_ids, p_ids = pair_to_arrays(s_clean, p_clean, tok, MAX_LEN)
    proba = float(model.predict([s_ids, p_ids], verbose=0)[0][0])
    return round(proba * 100, 2)

sus = st.text_area("Suspected text", height=200)
src = st.text_area("Single source text (optional)", height=200)
uploads = st.file_uploader("Or upload .txt/.pdf (multiple allowed)", type=["txt","pdf"], accept_multiple_files=True)

if st.button("Check"):
    if not sus.strip():
        st.warning("Paste suspected text.")
        st.stop()

    candidates = []
    if src.strip():
        candidates.append(("textarea_source", src))
    for f in uploads or []:
        txt = read_text_file(f) if f.type == "text/plain" else read_pdf_file(f)
        if txt.strip():
            candidates.append((f.name, txt))
    if CORPUS_DIR.exists():
        for p in CORPUS_DIR.glob("**/*"):
            if p.suffix.lower() not in {".txt", ".pdf"}:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore") if p.suffix==".txt" else ""
                if text.strip():
                    candidates.append((str(p), text))
            except Exception:
                pass
    if not candidates:
        candidates = [("self", sus)]

    rows = []
    for name, raw in candidates:
        for chunk in chunk_text(raw):
            pct = predict_local(chunk, sus)
            rows.append({
                "source": name,
                "chunk_preview": chunk[:160].replace("\n"," ") + ("…" if len(chunk) > 160 else ""),
                "plag_percent": pct
            })

    df = pd.DataFrame(rows).sort_values("plag_percent", ascending=False)
    st.subheader("Top matches")
    st.dataframe(df.head(20), use_container_width=True)
    if not df.empty:
        st.metric("Highest plagiarism likelihood", f"{df.iloc[0]['plag_percent']:.2f} %")

st.caption("Mode: **Local model** · File: output/lstm_model.keras")