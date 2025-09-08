# 🧠 LSTM Plagiarism Detector


## ⚙️ Setup

python -m venv .venv
# Activate venv
. .venv/Scripts/activate     # Windows
# or
source .venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📊 Run EDA

``` bash
python src/eda.py
```

Artifacts will be saved in `output/eda/`.

------------------------------------------------------------------------

## 🏋️ Train LSTM

``` bash
python src/train_lstm.py
```

Saves: - `output/lstm_model.keras` - `output/tokenizer.pkl` -
`output/lstm_metrics.json`

------------------------------------------------------------------------

## 🖥 Run Streamlit App (local)

``` bash
streamlit run app/streamlit_lstm.py
```

UI opens at <http://localhost:8501>.

