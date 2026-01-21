# 📚 Book Review Sentiment Classifier

A machine learning project that classifies book reviews into **POSITIVE**, **NEGATIVE**, or **NEUTRAL** sentiments using **scikit-learn**.


---

## 🚀 Project Overview

The goal of this project is to analyze textual book reviews and predict their sentiment based on the review content.

### Example Predictions

```text
"Quite a story. Hard to put down." → POSITIVE
"Useless book, no one should ever read it" → NEGATIVE
```

---

## 🎯 Why This Project?

* First hands-on experience with **scikit-learn**
* Learn the end-to-end ML workflow:

  * Data preprocessing
  * Feature extraction (NLP)
  * Model training and evaluation
  * Hyperparameter tuning
* Experiment with multiple classification algorithms and compare performance

---

## 🗂 Project Structure

```text
├── Books_small_10000.json   # Dataset containing 10,000 book reviews
├── sklearn.ipynb            # Jupyter notebook with full implementation
└── README.md                # Project documentation
```

---

## 📊 Dataset

* Source: [https://github.com/KeithGalli/sklearn](https://github.com/KeithGalli/sklearn)
* Contains **10,000 book reviews**
* Each review includes:

  * Review text
  * Rating (1–5 stars)

### Sentiment Mapping

| Rating | Sentiment |
| ------ | --------- |
| 1, 2   | NEGATIVE  |
| 3      | NEUTRAL   |
| 4, 5   | POSITIVE  |

---

## ⚙️ How It Works

### 1️⃣ Data Preparation

* **Train/Test Split**:

  * 67% training
  * 33% testing
* **Class Balancing**:

  * Each sentiment class is truncated to have an equal number of samples
  * Helps reduce bias during training

---

### 2️⃣ Text Vectorization

* **Bag of Words (BoW)** approach
* Uses **TF-IDF (Term Frequency–Inverse Document Frequency)**:

  * Converts text into numerical vectors
  * Assigns lower weights to commonly occurring words
  * Improves model performance and accuracy

---

## 🤖 Model Training & Evaluation

Three classification models were trained and evaluated using **mean accuracy** and **F1 scores**.

### Model Performance

| Model                | Mean Accuracy | F1 Scores (NEG, NEU, POS) |
| -------------------- | ------------- | ------------------------- |
| SVM                  | 0.4343        | [0.4679, 0.4009, 0.4390]  |
| Gaussian Naive Bayes | **0.5994**    | [0.6888, 0.5155, 0.5931]  |
| Decision Tree        | 0.4471        | [0.4931, 0.3990, 0.4467]  |

Despite accuracy differences, **SVM was selected** due to better overall balance and suitability for high-dimensional text data.

---

## 🔧 Hyperparameter Tuning (SVM)

Performed hyperparameter tuning on the Support Vector Machine classifier.

### Parameters Tested

* **Kernel**: `linear`, `rbf`
* **C**: `1, 2, 4, 8, 16`

### Best Parameters

```text
Kernel = rbf
C = 1
```

The tuned SVM model was saved as a pickle file for reuse.

---

## 🧠 Technologies Used

* Python
* scikit-learn
* Jupyter Notebook
* TF-IDF Vectorizer
* Pickle

---

## 🙌 Acknowledgements

This project was built as a hands-on introduction to the scikit-learn library, inspired by Keith Galli’s tutorial:
👉 *Real-World Python Machine Learning Tutorial w/ Scikit Learn*
[https://youtu.be/M9Itm95JzL0](https://youtu.be/M9Itm95JzL0)

---

