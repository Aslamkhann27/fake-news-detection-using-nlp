# Fake News Detection Using NLP

This project trains and serves a fake-news classifier using TF-IDF features and Logistic Regression.

## Features
- Train a model from Kaggle-style `Fake.csv` and `True.csv` datasets.
- Save model artifacts as:
  - `fake_news_model.pkl`
  - `tfidf_vectorizer.pkl`
- Run a simple Flask web app for real-time predictions.

## Project Structure
```text
fake-news-detection-using-nlp/
├── app.py
├── model.py
├── index.html
├── requirements.txt
├── data/
│   ├── Fake.csv
│   └── True.csv
├── fake_news_model.pkl
└── tfidf_vectorizer.pkl
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model
Default training source is `data/`:

```bash
python model.py
```

You can also pass a custom dataset path:

```bash
python model.py --csv /path/to/data_or_csv
```

## Run the Web App
```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

## Notes
- If model files are missing, run `python model.py` before using the app.
- Input text is cleaned by lowercasing, removing non-letters, and filtering stopwords while preserving negation words.
