# Humor Detection Using Natural Language Processing

**Author:** Sareena Bilal · [sb743915@gmail.com](mailto:sb743915@gmail.com)
**Repository:** https://github.com/Sareena-2001/HUMOR-DETECTION-USING-NATURAL-LANGUAGE-PROCESSING

A binary text classification project that decides whether a short English text is **humorous (1)** or **non-humorous (0)**. Texts are cleaned with a standard NLP pipeline, converted to TF-IDF features, and used to train and compare six classical machine learning classifiers.

---

## Results at a glance

Best model: **Support Vector Machine (RBF kernel)** on both datasets.

| Classifier | Accuracy (20k) | F1 (20k) | Accuracy (50k) | F1 (50k) |
|---|---|---|---|---|
| Multinomial Naive Bayes | 87.00% | 0.8735 | 88.17% | 0.8836 |
| Logistic Regression | 86.60% | 0.8679 | 87.54% | 0.8764 |
| Decision Tree | 76.75% | 0.7799 | 78.14% | 0.7916 |
| Random Forest | 82.63% | 0.8333 | 84.27% | 0.8480 |
| Gradient Boosting | 75.35% | 0.7310 | 74.66% | 0.7135 |
| **SVM (SVC)** | **87.25%** | **0.8741** | **88.80%** | **0.8890** |

Precision, recall and F1 are reported for the humorous class. The test sets contain 4,000 (20k) and 10,000 (50k) samples.

---

## Repository contents

| File | Description |
|---|---|
| `HUMOR_DETECTION.ipynb` | Main notebook: experiments on the 20k dataset |
| `HUMOR_DETECTION_PART2.ipynb` | Same pipeline on the 50k dataset |
| `humor_detection (1).py` | Script version of the 20k notebook |
| `part_2.py` | Script version of the 50k notebook |
| `processed_data(20k).csv` | 20,000 labeled short texts (`text`, `humour`) |
| `processed_data(50k).csv` | 50,000 labeled short texts (`text`, `humour`) |
| `archive.zip` | Compressed copy of both CSV files |

## Dataset

Each CSV has two columns:

- `text`: a short English sentence (already lowercased and stemmed, about 7 tokens on average)
- `humour`: label, `1` = humorous, `0` = non-humorous

Both datasets are balanced (about 50% per class) and have no missing values. Their content is consistent with subsets of the 200k short-text humor dataset of Annamoradnejad & Zoghi (ColBERT, 2020), where humorous samples are short jokes and non-humorous samples are mostly news headlines.

## Methodology

```
Raw texts ─► Preprocessing ─► TF-IDF ─► 80/20 split ─► 6 classifiers ─► Evaluation
```

1. **Preprocessing:** lowercase → remove URLs → keep letters only → remove NLTK stopwords → Porter stemming → WordNet lemmatization
2. **Features:** `TfidfVectorizer` (unigrams, default settings), fitted on the training set only
3. **Split:** `train_test_split(test_size=0.2, random_state=42)`
4. **Classifiers (scikit-learn defaults):** MultinomialNB, LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, SVC
5. **Metrics:** accuracy, precision, recall, F1-score, confusion matrix; best model selected by accuracy

## Getting started

### Requirements

- Python 3.8+
- pandas, numpy, nltk, scikit-learn, matplotlib, seaborn

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

### Run

```bash
git clone https://github.com/Sareena-2001/HUMOR-DETECTION-USING-NATURAL-LANGUAGE-PROCESSING.git
cd HUMOR-DETECTION-USING-NATURAL-LANGUAGE-PROCESSING
```

The scripts were exported from Google Colab and load data from `/content/sample_data/`. To run locally, change the path in the `pd.read_csv(...)` line, for example:

```python
dataset = pd.read_csv('processed_data(20k).csv')
```

Then run:

```bash
python "humor_detection (1).py"   # 20k dataset
python part_2.py                   # 50k dataset
```

Or open the notebooks in Jupyter or Google Colab. NLTK stopwords and WordNet are downloaded automatically on first run.

> Note: the SVM and Gradient Boosting models can take several minutes to train on the 50k dataset on a CPU.

## Output

For each classifier the scripts print a classification report, the confusion matrix and the F1, recall, accuracy and precision scores, and they plot:

- a confusion matrix heatmap per classifier
- a training versus testing accuracy bar chart
- a pie chart and bar chart of the label distribution

## Key findings

- Linear and probabilistic models (SVM, Naive Bayes, Logistic Regression) work best on sparse TF-IDF features of short texts.
- Decision Tree and Random Forest overfit (near-perfect training accuracy), while default Gradient Boosting underfits and has low recall for humor.
- More data (16k → 40k training samples) improves accuracy for 5 of the 6 models.
- Frequent non-humorous words are news-related (for example "trump", "photo", "video"), so part of the signal may come from source or topic differences rather than humor itself.

## Future work

- Hyperparameter tuning with cross-validation
- Word and character n-grams
- Word embeddings and fine-tuned transformers (BERT)
- Topic-controlled evaluation sets to reduce source bias
- Deployment as a simple web app or API

## Research paper

A full write-up of this project is available as `Humor_Detection_Research_Paper.docx` and `Humor_Detection_Research_Paper.pdf`.

## Citation

```
S. Bilal, "Humor Detection Using Natural Language Processing: A Comparative Study of
Classical Machine Learning Classifiers on Short Texts," 2026. [Online]. Available:
https://github.com/Sareena-2001/HUMOR-DETECTION-USING-NATURAL-LANGUAGE-PROCESSING
```

## Contact

Sareena Bilal · sb743915@gmail.com
