# Nepali Fake News Detection

A machine learning pipeline for detecting fake news in Nepali-language text. The project loads a multi-part Nepali news dataset, cleans and validates it, extracts TF-IDF text features, trains and compares six classifiers, and saves the best-performing model for reuse.

## Overview

News articles written in Nepali are classified as **Real (0)** or **Fake (1)** using classical supervised machine learning models trained on TF-IDF features. Six algorithms are trained under identical conditions so their performance can be directly compared, and the strongest model (by F1-score) is selected, evaluated in depth, and persisted to disk.

## Dataset

- **Source:** ## Dataset Setup

This project uses the [Nepali Fake News Detection dataset](https://www.kaggle.com/datasets/ashoknepal/nepali-fake-news-detection) from Kaggle.

1. Download the dataset from Kaggle (requires a free account).
2. Place the extracted CSV files (`nepali_news_part_*.csv`) into a local folder, e.g. `data/`.
3. Update the `data_path` variable in the notebook to point to that folder:
```python
   data_path = "data/"
```
The dataset is not included in this repository due to size and licensing — you must download it separately before running the notebook.
`nepali-fake-news-detection` dataset (Kaggle, by ashoknepal), distributed as multiple CSV parts (`nepali_news_part_*.csv`).

- **Expected columns:**
  | Column | Description |
  |---|---|
  | `news_context` | Nepali news article text (primary input feature) |
  | `label` | Binary target — `0` = Real, `1` = Fake |
  | `category` | News category |
  | `source_type` | Source of the news |
  | `news_id` | Unique news identifier |
  | `generated_at` | Timestamp |
  | `meta_intent` | Intended communication style |
  | `meta_style` | Writing style |

Only `news_context` and `label` are used for the baseline text classification model; the remaining metadata columns are dropped before training.

## Pipeline

1. **Data loading** — reads all CSV parts (malformed rows are skipped with a warning) and concatenates them into a single DataFrame.
2. **Data cleaning**
   - Drop rows with missing `label`
   - Cast `label` to integer and validate it only contains `{0, 1}`
   - Remove duplicate `news_context` records (keeping the first occurrence)
   - Check for empty or extremely short (<10 characters) text
3. **Exploratory analysis** — missing-value summary, label distribution (pie/bar charts), duplicate analysis, text length statistics.
4. **Train/test split** — 80/20 split, stratified on `label` (`random_state=42`).
5. **Feature extraction** — `TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)` fit on the training text only.
6. **Model training** — six classifiers trained on the same TF-IDF matrix:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM (`LinearSVC`)
   - Decision Tree
   - Random Forest (200 trees)
   - K-Nearest Neighbors (k=5)
7. **Evaluation** — Accuracy, Precision, Recall, and F1-score per model; classification report and confusion matrix for the best model; ROC-AUC curves across all models.
8. **Model selection** — best model chosen by highest F1-score (tie-broken by Accuracy).
9. **Error analysis** — inspection of misclassified samples, split into false positives (Real → Fake) and false negatives (Fake → Real).
10. **Persistence** — best model and fitted TF-IDF vectorizer saved with `joblib` for reuse without retraining.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **SVM** | **0.9712** | **0.9810** | **0.9595** | **0.9701** |
| Logistic Regression | 0.9652 | 0.9764 | 0.9517 | 0.9639 |
| Naive Bayes | 0.9616 | 0.9755 | 0.9451 | 0.9600 |
| Random Forest | 0.9592 | 0.9721 | 0.9434 | 0.9576 |
| KNN | 0.9447 | 0.9692 | 0.9158 | 0.9417 |
| Decision Tree | 0.9143 | 0.9141 | 0.9097 | 0.9119 |

**Best model: Linear SVM** — leads on every metric, making it the clear choice for this task.

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Usage

### 1. Train from scratch
Run the notebook `nepali-fake-news-detection.ipynb` top to bottom. It will load the data, clean it, train all six models, evaluate them, and save `best_model.pkl` and `tfidf_vectorizer.pkl` at the end.

### 2. Reuse the saved model (no retraining required)
```python
import joblib

best_model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

label_map = {0: "Real", 1: "Fake"}

new_text = ["काठमाण्डौमा आजको मौसम राम्रो छ।"]
new_text_tfidf = tfidf.transform(new_text)
prediction = best_model.predict(new_text_tfidf)

print(f"Prediction: {label_map[prediction[0]]}")
```

## Project Structure

```
.
├── nepali-fake-news-detection.ipynb   # Full analysis & training pipeline
├── best_model.pkl                     # Saved best-performing classifier (SVM)
├── tfidf_vectorizer.pkl               # Saved fitted TF-IDF vectorizer
└── README.md
```

## Notes & Limitations

- Text preprocessing is minimal (no stemming/lemmatization specific to Nepali morphology); results may improve with Nepali-specific NLP preprocessing.
- `LinearSVC` does not natively output probabilities; ROC-AUC uses `decision_function` scores for this model instead of `predict_proba`.
- The dataset's `category`, `source_type`, and other metadata fields are not used in the baseline model but could be incorporated as additional features in future work.
