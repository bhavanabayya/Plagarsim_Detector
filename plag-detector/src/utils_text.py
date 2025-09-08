from typing import Tuple
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_tokenizer(texts, vocab_size=30000, oov_token="<OOV>") -> Tokenizer:
    tok = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tok.fit_on_texts(texts)
    return tok

def pair_to_arrays(src_clean: str, sus_clean: str, tok: Tokenizer, max_len=60) -> Tuple[np.ndarray, np.ndarray]:
    s = pad_sequences(tok.texts_to_sequences([src_clean]), maxlen=max_len, padding="post", truncating="post")
    p = pad_sequences(tok.texts_to_sequences([sus_clean]), maxlen=max_len, padding="post", truncating="post")
    return s, p
