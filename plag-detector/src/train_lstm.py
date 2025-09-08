from pathlib import Path
import csv, json
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess import clean_text
from utils_text import build_tokenizer

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import joblib

DATA_TSV = Path("data/train_snli.txt")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

TOKENIZER_PATH = OUT / "tokenizer.pkl"
MODEL_DIR = OUT / "lstm_model"
METRICS_PATH = OUT / "lstm_metrics.json"

MAX_LEN = 60
VOCAB_SIZE = 30000
EMB_DIM = 128

def tsv_to_df(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3: continue
            s, p, y = parts
            try: y = int(y)
            except: continue
            rows.append((s, p, y))
    return pd.DataFrame(rows, columns=["source_txt","plagiarism_txt","label"])

def build_model(vocab_size, emb_dim, max_len) -> Model:
    inp1 = Input(shape=(max_len,), name="src_ids")
    inp2 = Input(shape=(max_len,), name="sus_ids")

    emb = Embedding(vocab_size, emb_dim, mask_zero=True, name="emb")
    lstm = Bidirectional(LSTM(96, return_sequences=False, name="bilstm"))

    h1 = lstm(emb(inp1))
    h2 = lstm(emb(inp2))

    # interaction features
    diff = Lambda(lambda t: tf.abs(t[0]-t[1]))([h1, h2])
    mult = Lambda(lambda t: t[0]*t[1])([h1, h2])
    x = Concatenate()([h1, h2, diff, mult])
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model([inp1, inp2], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def main():
    assert DATA_TSV.exists(), f"Put train_snli.txt in {DATA_TSV}"
    print("[1/5] Load & clean")
    df = tsv_to_df(DATA_TSV).dropna().drop_duplicates()

    tqdm.pandas()
    df["src_p"]  = df["source_txt"].progress_apply(clean_text)
    df["sus_p"]  = df["plagiarism_txt"].progress_apply(clean_text)

    print("[2/5] Tokenizer")
    tok = build_tokenizer(pd.concat([df["src_p"], df["sus_p"]]).tolist(), vocab_size=VOCAB_SIZE)
    joblib.dump(tok, TOKENIZER_PATH)

    from utils_text import pair_to_arrays
    def to_arrays_batch(frame: pd.DataFrame):
        src_ids = tok.texts_to_sequences(frame["src_p"])
        sus_ids = tok.texts_to_sequences(frame["sus_p"])
        src_pad = tf.keras.preprocessing.sequence.pad_sequences(src_ids, maxlen=MAX_LEN, padding="post", truncating="post")
        sus_pad = tf.keras.preprocessing.sequence.pad_sequences(sus_ids, maxlen=MAX_LEN, padding="post", truncating="post")
        return src_pad, sus_pad

    print("[3/5] Train/val split")
    from sklearn.model_selection import train_test_split
    X_src, X_sus = to_arrays_batch(df)
    y = df["label"].to_numpy().astype("float32")
    s_tr, s_te, p_tr, p_te, y_tr, y_te = train_test_split(X_src, X_sus, y, test_size=0.2, random_state=42)

    print("[4/5] Build & train LSTM")
    model = build_model(VOCAB_SIZE, EMB_DIM, MAX_LEN)
    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    model.fit([s_tr, p_tr], y_tr,
              validation_data=([s_te, p_te], y_te),
              epochs=3, batch_size=256, callbacks=[es])

    print("[5/5] Evaluate & save")
    loss, acc = model.evaluate([s_te, p_te], y_te, verbose=0)

    METRICS_PATH.write_text(
        json.dumps({"accuracy": float(acc), "loss": float(loss)}, indent=2),
        encoding="utf-8"
    )

    MODEL_FILE = OUT / "lstm_model.keras"   # <- NEW: file instead of directory
    model.save(MODEL_FILE, include_optimizer=False)

    print(f"Saved model -> {MODEL_FILE}\nSaved tokenizer -> {TOKENIZER_PATH}\nAcc={acc:.4f}")


if __name__ == "__main__":
    main()
