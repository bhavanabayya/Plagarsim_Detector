# src/train_lstm.py
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# --- make package imports work when run directly ---
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocess import clean_text  # your existing cleaner

# Keras / TF (Keras 3)
import tensorflow as tf
from keras import layers, Model, optimizers, callbacks
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split
import joblib


# ------------------------ Paths & Config ------------------------
DATA_TXT   = ROOT / "data" / "train_snli.txt"     # src \t plag \t label
OUT        = ROOT / "output"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_FILE     = OUT / "lstm_model.keras"         # Keras 3 native format
TOKENIZER_PATH = OUT / "tokenizer.pkl"
METRICS_PATH   = OUT / "lstm_metrics.json"

# Training hyperparams
MAX_LEN        = 60
VOCAB_SIZE     = 50000
EMBED_DIM      = 128
LSTM_UNITS     = 128
DENSE_UNITS    = 128
DROPOUT        = 0.3
BATCH_SIZE     = 256
EPOCHS         = 6
VAL_SIZE       = 0.2
RANDOM_STATE   = 42

# ---------------------------------------------------------------


def read_snli_tsv(path: Path) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Assumes each line: source<TAB>plagiarism<TAB>label
    """
    srcs, plags, labels = [], [], []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            s, p, y = parts
            srcs.append(s)
            plags.append(p)
            try:
                labels.append(int(y))
            except Exception:
                continue
    return srcs, plags, np.array(labels, dtype=np.int32)


def prepare_texts(srcs: List[str], plags: List[str]) -> Tuple[List[str], List[str]]:
    print("[1/5] Load & clean")
    srcs_clean = [clean_text(s or "") for s in tqdm(srcs)]
    plags_clean = [clean_text(p or "") for p in tqdm(plags)]
    return srcs_clean, plags_clean


def build_tokenizer(corpus: List[str], vocab_size: int) -> Tokenizer:
    print("[2/5] Tokenizer")
    tok = Tokenizer(num_words=vocab_size, oov_token="<OOV>", filters="")
    tok.fit_on_texts(corpus)
    return tok


def to_arrays(
    tok: Tokenizer, srcs: List[str], plags: List[str], max_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    print("[3/5] Train/val split & vectorize")
    s_ids = tok.texts_to_sequences(srcs)
    p_ids = tok.texts_to_sequences(plags)
    s_ids = pad_sequences(s_ids, maxlen=max_len, padding="post", truncating="post")
    p_ids = pad_sequences(p_ids, maxlen=max_len, padding="post", truncating="post")
    return s_ids.astype("int32"), p_ids.astype("int32")


def build_model(
    vocab_size: int,
    embed_dim: int,
    lstm_units: int,
    dense_units: int,
    dropout: float,
    max_len: int,
) -> Model:
    """
    Two-tower shared Embedding + BiLSTM encoder.
    SAFE combine: Subtract -> abs (single-tensor Lambda), Concatenate, Dot(normalize=True)
    """
    # Inputs
    src_ids = layers.Input(shape=(max_len,), dtype="int32", name="src_ids")
    plag_ids = layers.Input(shape=(max_len,), dtype="int32", name="plag_ids")

    # Shared embedding & encoder
    embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True, name="embed")
    encoder = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False), name="bilstm")

    # Encode both sides
    src_emb = embedding(src_ids)
    plag_emb = embedding(plag_ids)
    enc_src = encoder(src_emb)     # (None, 2*lstm_units)
    enc_plg = encoder(plag_emb)

    # Feature engineering (no unsafe Lambda over list)
    diff = layers.Subtract(name="diff")([enc_src, enc_plg])     # (None, 2*lstm_units)
    abs_diff = layers.Lambda(tf.abs, name="abs")(diff)          # safe single-tensor lambda
    cos = layers.Dot(axes=-1, normalize=True, name="cosine")([enc_src, enc_plg])  # (None, 1)

    feat = layers.Concatenate(name="feat")([enc_src, enc_plg, abs_diff, cos])

    x = layers.Dense(dense_units, activation="relu")(feat)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units // 2, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="prob")(x)

    model = Model(inputs=[src_ids, plag_ids], outputs=out, name="plag_lstm")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=2e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    # Reproducibility
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # 0) Read raw
    src_raw, plag_raw, labels = read_snli_tsv(DATA_TXT)

    # 1) Clean
    src_clean, plag_clean = prepare_texts(src_raw, plag_raw)

    # 2) Tokenizer (fit on combined corpus for shared vocab)
    tok = build_tokenizer(src_clean + plag_clean, VOCAB_SIZE)

    # 3) Vectorize
    s_ids, p_ids = to_arrays(tok, src_clean, plag_clean, MAX_LEN)

    # Split
    s_tr, s_te, p_tr, p_te, y_tr, y_te = train_test_split(
        s_ids, p_ids, labels, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    # 4) Build model
    print("[4/5] Build & train LSTM")
    model = build_model(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        lstm_units=LSTM_UNITS,
        dense_units=DENSE_UNITS,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    )

    es = callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    rp = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1)

    model.fit(
        [s_tr, p_tr],
        y_tr,
        validation_data=([s_te, p_te], y_te),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[es, rp],
    )

    # 5) Evaluate & save
    print("[5/5] Evaluate & save")
    loss, acc = model.evaluate([s_te, p_te], y_te, verbose=0)

    # Save artifacts
    joblib.dump(tok, TOKENIZER_PATH)
    model.save(MODEL_FILE, include_optimizer=False)

    METRICS_PATH.write_text(
        json.dumps({"accuracy": float(acc), "loss": float(loss)}, indent=2),
        encoding="utf-8",
    )

    print("Artifacts:")
    print(f" - {MODEL_FILE}")
    print(f" - {TOKENIZER_PATH}")
    print(f" - {METRICS_PATH}")
    print(f"Val Acc = {acc:.4f}, Val Loss = {loss:.4f}")


if __name__ == "__main__":
    # Optional: make TF a bit quieter
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
