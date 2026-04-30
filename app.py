import pickle
import re
from pathlib import Path

from flask import Flask, render_template_string, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

MODEL_PATH = Path("fake_news_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
HTML_PATH = Path("index.html")


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    preserved_words = {"no", "nor", "not", "never", "against", "without"}
    stopwords = ENGLISH_STOP_WORDS.difference(preserved_words)
    words = [word for word in text.split() if word not in stopwords]
    return " ".join(words)


def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


model, vectorizer = load_artifacts()


def load_html_template():
    if not HTML_PATH.exists():
        return "<h1>index.html not found</h1>"
    return HTML_PATH.read_text(encoding="utf-8")

@app.route("/", methods=["GET", "POST"])
def index():
    global model, vectorizer

    if model is None or vectorizer is None:
        model, vectorizer = load_artifacts()

    notice = None
    prediction = None
    prediction_class = ""
    confidence = None
    news_title = ""
    news_text = ""

    if model is None or vectorizer is None:
        notice = "Model files are missing. Run model.py first to train and save the artifacts."

    if request.method == "POST":
        news_title = request.form.get("news_title", "").strip()
        news_text = request.form.get("news_text", "").strip()
        if not news_title and not news_text:
            notice = "Please enter some news content to analyze."
        elif model is not None and vectorizer is not None:
            combined_text = f"{news_title} {news_text}".strip()
            processed_text = preprocess_text(combined_text)
            features = vectorizer.transform([processed_text])
            probability = model.predict_proba(features)[0]
            threshold = float(getattr(model, "threshold_", 0.5))
            prediction_value = 1 if probability[1] >= threshold else 0
            prediction = "REAL" if prediction_value == 1 else "FAKE"
            prediction_class = "real" if prediction == "REAL" else "fake"
            confidence = round(float(max(probability)) * 100, 2)

    return render_template_string(
        load_html_template(),
        notice=notice,
        prediction=prediction,
        prediction_class=prediction_class,
        confidence=confidence,
        news_title=news_title,
        news_text=news_text,
    )


if __name__ == "__main__":
    app.run(debug=True)
