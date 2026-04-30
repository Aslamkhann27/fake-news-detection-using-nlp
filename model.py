import argparse
import pickle
import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_PATH = Path("fake_news_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")
PRESERVED_WORDS = {"no", "nor", "not", "never", "against", "without"}
STOPWORDS = ENGLISH_STOP_WORDS.difference(PRESERVED_WORDS)


def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(words)


def build_source_text(data):
    if "title" in data.columns:
        title = data["title"].fillna("").astype(str)
        body = data["text"].fillna("").astype(str)
        combined = (title + " " + body).str.strip()
        data = data.copy()
        data["source_text"] = combined.where(combined != "", body)
    else:
        data = data.copy()
        data["source_text"] = data["text"].astype(str)
    return data


def load_dataset(source_path):
    source = Path(source_path)

    if source.is_dir():
        fake_path = source / "Fake.csv"
        real_path = source / "True.csv"

        if not fake_path.exists() or not real_path.exists():
            raise FileNotFoundError(
                "Expected Fake.csv and True.csv inside the dataset folder."
            )

        fake_data = pd.read_csv(fake_path)
        real_data = pd.read_csv(real_path)

        fake_data = fake_data[[c for c in ["title", "text"] if c in fake_data.columns]].dropna(subset=["text"]).copy()
        real_data = real_data[[c for c in ["title", "text"] if c in real_data.columns]].dropna(subset=["text"]).copy()
        fake_data["label"] = "FAKE"
        real_data["label"] = "REAL"

        data = pd.concat([fake_data, real_data], ignore_index=True)
    else:
        data = pd.read_csv(source)
        required_columns = {"text", "label"}
        if not required_columns.issubset(data.columns):
            raise ValueError("CSV file must contain 'text' and 'label' columns.")

        available_columns = [column for column in ["title", "text", "label"] if column in data.columns]
        data = data[available_columns].dropna(subset=["text"]).copy()
        data["label"] = data["label"].astype(str).str.strip().str.upper()
        data = data[data["label"].isin(["REAL", "FAKE"])]

    data = build_source_text(data)
    data["clean_text"] = data["source_text"].apply(preprocess_text)
    data["target"] = data["label"].map({"FAKE": 0, "REAL": 1})
    return data


def train_model(csv_path):
    data = load_dataset(csv_path)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data["clean_text"],
        data["target"],
        test_size=0.2,
        random_state=42,
        stratify=data["target"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.125,
        random_state=42,
        stratify=y_train_full,
    )

    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        C=2.0,
    )
    model.fit(X_train_tfidf, y_train)

    val_probabilities = model.predict_proba(X_val_tfidf)[:, 1]
    best_threshold = 0.5
    best_accuracy = 0.0

    for threshold in [x / 100 for x in range(35, 66)]:
        val_predictions = (val_probabilities >= threshold).astype(int)
        val_accuracy = accuracy_score(y_val, val_predictions)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_threshold = threshold

    test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]
    predictions = (test_probabilities >= best_threshold).astype(int)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)
    model.threshold_ = best_threshold

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(VECTORIZER_PATH, "wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"Decision threshold: {best_threshold:.2f}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a fake news detection model.")
    parser.add_argument(
        "--csv",
        default="data",
        help="Path to the dataset CSV file or the Kaggle dataset folder containing Fake.csv and True.csv.",
    )
    args = parser.parse_args()
    train_model(args.csv)
