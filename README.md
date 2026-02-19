# Twitter-Sentiment-Analysis
Developed a robust sentiment analysis system for social media data (tweets) using Python and scikit-learn

Project Overview
This project focuses on building and evaluating machine learning models for sentiment analysis of social media text, specifically tweets. The goal is to classify tweets as either positive or negative, providing insights into public sentiment.

Dataset
The dataset used for this project is a large collection of tweets, pre-processed to include sentiment polarity and the tweet text. Key characteristics:

Size: 1.6 million tweets
Labels: 0 (negative) and 1 (positive)
Encoding: latin-1 (due to character set compatibility issues)
Methodology
1. Data Loading and Preprocessing
The raw dataset was loaded using pandas.
Irrelevant columns were dropped, retaining only polarity and text.
A clean_text column was created by converting all text to lowercase.
2. Feature Extraction
TF-IDF Vectorization: Text data was transformed into numerical features using TfidfVectorizer from scikit-learn.
Parameters: max_features=5000 and ngram_range=(1,2) were used to capture important unigrams and bigrams.
3. Model Training and Evaluation
The data was split into training (80%) and testing (20%) sets. Three different classification models were trained and evaluated:

a) Bernoulli Naive Bayes (BernoulliNB)
Accuracy: ~76.65%
F1-Score (macro avg): ~0.77
b) Linear Support Vector Classifier (LinearSVC)
Accuracy: ~79.53%
F1-Score (macro avg): ~0.80
c) Logistic Regression (LogisticRegression)
Accuracy: ~79.54%
F1-Score (macro avg): ~0.80
4. Sample Predictions
The trained models were used to predict sentiment on new, unseen tweet samples, demonstrating their practical application.

Technologies Used
Python
pandas: For data manipulation and analysis.
scikit-learn: For machine learning models (TF-IDF, BernoulliNB, LinearSVC, LogisticRegression) and evaluation metrics.
Setup and Installation
To run this project, ensure you have the following Python libraries installed:

pip install pandas scikit-learn
Usage
Download the dataset (e.g., data.csv) and place it in the project directory.
Run the provided Python notebook or script (.ipynb or .py).
The notebook will perform data loading, preprocessing, model training, evaluation, and demonstrate sample predictions.
