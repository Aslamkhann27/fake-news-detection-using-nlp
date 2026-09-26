# Fake News Detection Using NLP

A machine-learning web application that classifies news articles as **REAL** or **FAKE** using Natural Language Processing (NLP). The project trains a Logistic Regression classifier on TF-IDF features and exposes the model through a clean Flask web interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Running the Web App](#running-the-web-app)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Details](#model-details)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Misinformation spreads rapidly across the internet. This project provides a practical NLP pipeline to detect fake news by analysing the linguistic patterns of news content. A user can paste any news article (or just its headline) into the web interface and receive an instant **REAL / FAKE** verdict along with a confidence score.

---

## Features

- **Text preprocessing** — lowercasing, punctuation removal, and custom stop-word filtering that preserves negation words (`not`, `never`, `nor`, `no`, `against`, `without`).
- **TF-IDF vectorisation** — unigrams and bigrams with sublinear term-frequency scaling (`max_features=30 000`).
- **Logistic Regression classifier** — trained with class-weight balancing to handle imbalanced datasets.
- **Threshold tuning** — the optimal decision threshold is selected on a held-out validation set (searched from 0.35 to 0.65), improving accuracy over the default 0.5 cut-off.
- **Flexible dataset loader** — accepts either a single CSV file (must contain `text` and `label` columns) or the standard Kaggle folder layout (`Fake.csv` + `True.csv`).
- **Flask web interface** — responsive single-page app; works on desktop and mobile.
- **Confidence score** — the prediction is accompanied by the model's probability estimate.
- **Pre-trained artefacts** — `fake_news_model.pkl` and `tfidf_vectorizer.pkl` are included so you can run the app immediately without retraining.

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| Language | Python 3.8+ |
| Web framework | Flask |
| NLP / vectorisation | scikit-learn (`TfidfVectorizer`) |
| Classifier | scikit-learn (`LogisticRegression`) |
| Data handling | pandas |
| Model serialisation | pickle (stdlib) |
| Front-end | HTML5 / CSS3 (served via `render_template_string`) |

---

## Project Structure

```
fake-news-detection-using-nlp/
│
├── data/                        # Dataset folder (Kaggle layout)
│   ├── Fake.csv                 # Fake news articles
│   └── True.csv                 # Real news articles
│
├── app.py                       # Flask web application
├── model.py                     # Model training & evaluation script
├── index.html                   # Jinja2 HTML template for the UI
│
├── fake_news_model.pkl          # Saved Logistic Regression model
├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│
├── True.csv                     # Sample real-news data (root copy)
└── README.md                    # This file
```

---

## Dataset

The project is designed around the popular **Fake and Real News** dataset available on Kaggle:

> **Kaggle**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset consists of two CSV files:

| File | Label | Description |
|---|---|---|
| `Fake.csv` | FAKE | ~23 000 fake news articles |
| `True.csv` | REAL | ~21 000 real news articles |

Each file contains four columns: `title`, `text`, `subject`, and `date`. Only `title` and `text` are used by the model.

**Alternative format** — you can also supply a single CSV file that already has `text` and `label` columns (label values must be `REAL` or `FAKE`).

---

## Installation

### Prerequisites

- Python 3.8 or higher
- `pip`

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/Aslamkhann27/fake-news-detection-using-nlp.git
   cd fake-news-detection-using-nlp
   ```

2. **Create and activate a virtual environment** *(recommended)*

   ```bash
   python -m venv venv
   # Linux / macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install flask scikit-learn pandas
   ```

4. **Add the dataset** *(skip this step if you only want to use the pre-trained artefacts)*

   Download the Kaggle dataset and place `Fake.csv` and `True.csv` inside the `data/` folder:

   ```
   data/
   ├── Fake.csv
   └── True.csv
   ```

---

## Training the Model

Run `model.py` to train the classifier and save the model artefacts.

### Using the default `data/` folder

```bash
python model.py
```

### Using a custom dataset folder

```bash
python model.py --csv path/to/dataset/folder
```

### Using a single CSV file

```bash
python model.py --csv path/to/news_data.csv
```

The script will print training progress and final metrics:

```
Model saved to: fake_news_model.pkl
Vectorizer saved to: tfidf_vectorizer.pkl
Decision threshold: 0.48
Accuracy: 0.9923
Confusion Matrix:
[[4678   32]
 [  29 4235]]
```

> **Note:** The pre-trained `fake_news_model.pkl` and `tfidf_vectorizer.pkl` included in the repository let you skip this step entirely.

---

## Running the Web App

```bash
python app.py
```

The Flask development server starts on `http://127.0.0.1:5000` by default. Open that URL in your browser.

> **Production deployment:** For production use, serve the app with a WSGI server such as **Gunicorn**:
>
> ```bash
> pip install gunicorn
> gunicorn -w 2 app:app
> ```

---

## Usage

1. Open `http://127.0.0.1:5000` in your browser.
2. Paste the **news article text** (and optionally the headline) into the text area.
3. Click **Predict**.
4. The result panel shows:
   - **REAL** (green) or **FAKE** (red) label.
   - **Confidence** — the model's probability score as a percentage.

---

## How It Works

```
Raw news text
      │
      ▼
┌─────────────────────────────┐
│  Text Preprocessing          │
│  • lowercase                 │
│  • remove non-alpha chars    │
│  • remove stop-words         │
│    (preserve negations)      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  TF-IDF Vectorisation        │
│  • unigrams + bigrams        │
│  • sublinear TF scaling      │
│  • max 30 000 features       │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Logistic Regression         │
│  • class-weight balanced     │
│  • tuned decision threshold  │
└────────────┬────────────────┘
             │
             ▼
      REAL  /  FAKE
   + confidence score
```

### Why preserve negation words?

Standard stop-word lists remove words like *not* and *never*, which can flip the sentiment of a sentence. Keeping these words allows the model to distinguish between "the government **did** act" and "the government **did not** act".

---

## Model Details

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression |
| Solver | `liblinear` |
| Regularisation (`C`) | `2.0` |
| Max iterations | `2000` |
| Class weight | `balanced` |
| TF-IDF features | `30 000` |
| N-gram range | `(1, 2)` |
| Sublinear TF | `True` |
| Train / val / test split | 70 % / 10 % / 20 % |
| Threshold tuning range | 0.35 – 0.65 (step 0.01) |

The decision threshold is tuned on the validation split to maximise accuracy, then the final model is evaluated on the held-out test split.

---

## Screenshots

> *(Add screenshots here once the application is running.)*

**Home page – input form**

```
┌──────────────────────────────────────┐
│         Fake News Detection          │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ Enter news content here...     │  │
│  │                                │  │
│  └────────────────────────────────┘  │
│                                      │
│          [ Predict ]                 │
└──────────────────────────────────────┘
```

**Result – REAL news**

```
┌──────────────────────────────────────┐
│    Prediction: REAL                  │  ← green
│    Confidence: 97.43%                │
└──────────────────────────────────────┘
```

**Result – FAKE news**

```
┌──────────────────────────────────────┐
│    Prediction: FAKE                  │  ← red
│    Confidence: 88.61%                │
└──────────────────────────────────────┘
```

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature-name`.
3. Commit your changes: `git commit -m "Add your feature"`.
4. Push to the branch: `git push origin feature/your-feature-name`.
5. Open a Pull Request.

Please ensure your code follows the existing style and includes relevant tests or examples where applicable.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
