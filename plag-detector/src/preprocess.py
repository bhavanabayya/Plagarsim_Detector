import re, nltk, spacy, contractions
from nltk.corpus import stopwords

_nlp = None
_stop = None
_emoji_re = re.compile("[" +
    "\U0001F600-\U0001F64F"+"\U0001F300-\U0001F5FF"+"\U0001F680-\U0001F6FF"+
    "\U0001F700-\U0001F77F"+"\U0001F780-\U0001F7FF"+"\U0001F800-\U0001F8FF"+
    "\U0001F900-\U0001F9FF"+"\U0001FA00-\U0001FA6F"+"\U0001FA70-\U0001FAFF"+
    "\u2702-\u27B0"+"\u24C2-\U0001F251"+"]", flags=re.UNICODE)

def _ensure():
    global _nlp, _stop
    if _stop is None:
        nltk.download("stopwords", quiet=True)
        _stop = set(stopwords.words("english"))
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner","parser"])

def clean_text(text: str) -> str:
    _ensure()
    if not isinstance(text, str): return ""
    text = contractions.fix(text.lower())
    text = re.sub(r"<.*?>","", text)
    text = re.sub(r"http\S+|www\S+|https\S+","", text)
    text = _emoji_re.sub("", text)
    text = re.sub(r"[^\w\s]"," ", text)
    text = re.sub(r"[^a-z\s]"," ", text)
    text = re.sub(r"\s+"," ", text).strip()
    tokens = [t.lemma_ for t in _nlp(text) if t.text not in _stop]
    return " ".join(tokens)
