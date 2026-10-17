# 🍔 Snapfood Sentiment Analysis

**Persian (Farsi) binary sentiment classification** on customer reviews from [Snapfood](https://snappfood.ir), Iran’s leading food-delivery platform.

Comments are classified as **HAPPY** or **SAD** using classic NLP techniques: TF-IDF vectorization + Linear Support Vector Classifier.

---

## 📊 Results at a Glance

| Metric              | Value      |
|---------------------|------------|
| Dataset size        | 52,110 reviews |
| Classes             | HAPPY (0) · SAD (1) |
| Train / Test split  | 80% / 20% (stratified) |
| Model               | `TfidfVectorizer` + `LinearSVC` |
| **Test Accuracy**   | **≈ 84.0%** |
| Macro F1-score      | 0.84 |

```
              precision    recall  f1-score   support

       HAPPY       0.86      0.81      0.84      5247
         SAD       0.82      0.87      0.84      5175

    accuracy                           0.84     10422
```

---

## 🗂️ Project Structure

```
.
├── main.ipynb              # Complete experiment notebook
├── snapfood_train.csv      # Dataset (comment, label, label_id)
└── README.md
```

---

## 📁 Dataset

| Column     | Description                          |
|------------|--------------------------------------|
| `comment`  | Customer review text (Persian)       |
| `label`    | Human-readable class (`HAPPY`/`SAD`) |
| `label_id` | Numeric label (`0` = HAPPY, `1` = SAD) |

**Sample reviews:**

| Comment | Label |
|---------|-------|
| غذا خیلی سرد بود در صورتیکه فاصله ما خیلی کم است | SAD |
| یکی از بهترین ته چین هاییه که خوردم | HAPPY |
| خیلی تازه و خوشمزه بود. بسته بندی شیک و عالی | HAPPY |
| غذا بد بود حالم خیییییلی بده | SAD |

---

## ⚙️ Method

1. **Data loading** – Read `snapfood_train.csv`
2. **Train/Test split** – Stratified 80/20 split (`random_state=42`)
3. **Feature extraction** – `TfidfVectorizer` (default word-level TF-IDF)
4. **Classification** – `LinearSVC` (scikit-learn defaults)
5. **Evaluation** – Confusion matrix, classification report, accuracy

> Earlier experiments also explored higher-order n-grams (`ngram_range=(1,8)`), which produce a very high-dimensional sparse feature space.

---

## 🚀 Quick Start

### Requirements

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### Run the notebook

1. Place `snapfood_train.csv` in the same folder as the notebook.
2. Open `main.ipynb` and run all cells.

### Example prediction

```python
test = "راضی نبودم"
test_transformed = tfidf.transform([test])
prediction = model.predict(test_transformed)

print(prediction)   # → [1.]  (SAD)
```

---

## 📈 Confusion Matrix

The notebook generates a heatmap of the confusion matrix on the held-out test set so you can quickly inspect true positives / false positives for each class.

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** – data handling
- **scikit-learn** – TF-IDF + LinearSVC
- **matplotlib / seaborn** – visualization

---

## 📝 Notes

- The model is intentionally simple and fast — suitable as a strong baseline for Persian sentiment analysis on food-delivery reviews.
- No external Persian NLP libraries (e.g. Hazm) are required; the pipeline works with raw text.
- Results may vary slightly depending on scikit-learn version and exact data split.

---

## 📄 License

Educational / research use. Please respect the original data source terms if you redistribute the dataset.

---

**Made for experimenting with classical NLP on real-world Persian customer feedback.**
