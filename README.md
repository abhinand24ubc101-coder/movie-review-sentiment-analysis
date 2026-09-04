# movie-review-sentiment-analysis
Movie Review Sentiment Analysis using TF-IDF and Linear SVM
# Movie Review Sentiment Analysis

## Project Overview

This project implements a Movie Review Sentiment Analysis system using Natural Language Processing and Machine Learning techniques.

The system classifies movie reviews into two categories:

* Positive
* Negative

TF-IDF is used to convert text into numerical features, and a Linear Support Vector Machine (SVM) is used for sentiment classification.

## Objectives

* Load and explore the movie review dataset.
* Check for missing values and duplicate reviews.
* Clean and preprocess movie review text.
* Remove HTML tags, URLs, special characters, and stopwords.
* Convert text into numerical features using TF-IDF.
* Train a Linear SVM classification model.
* Predict the sentiment of new movie reviews.
* Evaluate the model using Accuracy, Precision, Recall, and F1-Score.
* Generate a confusion matrix and classification report.
* Save the trained model and TF-IDF vectorizer.

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

## Dataset

The project uses a movie review dataset containing review text and sentiment labels.

The two sentiment classes are:

* Negative
* Positive

The dataset is loaded into a Pandas DataFrame and processed before training the machine learning model.

The dataset is not included in this repository if it is large. It can be uploaded to Google Colab before running the notebook.

## Project Workflow

1. Load the dataset
2. Explore the dataset
3. Check missing values and duplicates
4. Clean the review text
5. Split the dataset into training and testing data
6. Apply TF-IDF vectorization
7. Train the Linear SVM model
8. Generate predictions
9. Evaluate the model
10. Generate the confusion matrix
11. Test new movie reviews
12. Save the trained model and vectorizer

## Text Preprocessing

The following preprocessing steps are performed:

* Convert text to lowercase
* Remove HTML tags
* Remove URLs
* Remove special characters
* Remove stopwords
* Generate cleaned review text

Example:

```text
Original:
The movie was <b>AMAZING!</b> I really enjoyed it.

Cleaned:
movie amazing really enjoyed
```

## TF-IDF Vectorization

TF-IDF is used to convert the cleaned movie reviews into numerical feature vectors.

The vectorizer uses:

```python
TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)
```

The configuration uses unigrams and bigrams and limits the maximum number of extracted features to 100,000.

## Machine Learning Model

A Linear Support Vector Machine is used for sentiment classification.

```python
from sklearn.svm import LinearSVC

model = LinearSVC(random_state=42)

model.fit(X_train_tfidf, y_train)
```

## Model Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Classification Report
* Confusion Matrix

Add the actual results from the notebook below:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.XXXX |
| Precision | 0.XXXX |
| Recall    | 0.XXXX |
| F1-Score  | 0.XXXX |

## Sample Prediction

Example positive review:

```text
The movie was fantastic, emotional and beautifully acted.
```

Predicted sentiment:

```text
Positive
```

Example negative review:

```text
The film was disappointing and felt like a waste of time.
```

Predicted sentiment:

```text
Negative
```

## User Input Prediction

The project allows users to enter their own movie review and receive a predicted sentiment.

```python
user_review = input("Enter a movie review: ")

result = predict_sentiment(user_review)

print("\nPredicted Sentiment:", result)
```

## Saved Models

The trained model and TF-IDF vectorizer are saved using Joblib.

```text
models/
├── movie_sentiment_svm_model.pkl
└── movie_sentiment_tfidf_vectorizer.pkl
```

## Project Structure

```text
movie-review-sentiment-analysis/
│
├── Movie_Review_Sentiment_Analysis.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   ├── movie_sentiment_svm_model.pkl
│   └── movie_sentiment_tfidf_vectorizer.pkl
│
└── screenshots/
    ├── 01_dataset.png
    ├── 02_preprocessing.png
    ├── 03_tfidf.png
    ├── 04_model_training.png
    ├── 05_classification_report.png
    ├── 06_confusion_matrix.png
    ├── 07_predictions.png
    └── 08_final_performance.png
```

## Installation

Install the required libraries using:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/your-username/movie-review-sentiment-analysis.git
```

2. Open the notebook in Google Colab or Jupyter Notebook.

3. Upload the movie review dataset.

4. Update the dataset path in the notebook.

5. Run the notebook cells sequentially.

## Future Improvements

* Compare Linear SVM with other machine learning algorithms.
* Use Word2Vec or GloVe embeddings.
* Implement LSTM or Transformer-based models.
* Develop a web application for real-time sentiment prediction.
* Deploy the trained model as an API.

## Author

Abhinand Ajikumar
