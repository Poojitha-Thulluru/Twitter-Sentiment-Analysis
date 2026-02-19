# Twitter-Sentiment-Analysis
Developed a robust sentiment analysis system for social media data (tweets) using Python and scikit-learn

## Project Overview
This project focuses on building and evaluating machine learning models for sentiment analysis of social media text, specifically tweets. The goal is to classify tweets as either positive or negative, providing insights into public sentiment.

## Dataset
The dataset used for this project is a large collection of tweets, pre-processed to include sentiment polarity and the tweet text. Key characteristics:
- **Size:** 1.6 million tweets
- **Labels:** 0 (negative) and 4 (positive)
- **Encoding:** `latin-1` (due to character set compatibility issues)

## Methodology

### 1. Data Loading and Preprocessing
- The raw dataset was loaded using pandas.
- Irrelevant columns were dropped, retaining only `polarity` and `text`.
- A `clean_text` column was created by converting all text to lowercase.

### 2. Feature Extraction
- **TF-IDF Vectorization:** Text data was transformed into numerical features using `TfidfVectorizer` from `scikit-learn`.
- **Parameters:** `max_features=5000` to select the most frequently occurring terms, and `ngram_range=(1,2)` to include both single words (unigrams) and two-word phrases (bigrams) as features. This helps capture more context from the text.

### 3. Model Training and Evaluation
The data was split into training (80%) and testing (20%) sets using `random_state=42` for reproducibility. Three different classification models were trained and evaluated on the TF-IDF vectorized data:

#### a) Bernoulli Naive Bayes (`BernoulliNB`)
- **Description:** A probabilistic classifier based on Bayes' theorem, particularly suited for discrete features like word counts or the presence/absence of terms in a document. It assumes that features are conditionally independent given the class label.
- **Accuracy:** ~76.65%
- **F1-Score (macro avg):** ~0.77

#### b) Linear Support Vector Classifier (`LinearSVC`)
- **Description:** A linear classifier that attempts to find a hyperplane that best separates the classes in the high-dimensional feature space. It's an efficient implementation of Support Vector Machines for linear classification.
- **Parameters:** `max_iter=1000` was set to limit the number of iterations for the solver, ensuring convergence within a reasonable time.
- **Accuracy:** ~79.53%
- **F1-Score (macro avg):** ~0.80

#### c) Logistic Regression (`LogisticRegression`)
- **Description:** A linear model used for binary classification. Despite its name, it's a classification algorithm that models the probability of a binary outcome. It's widely used for its interpretability and efficiency.
- **Parameters:** `max_iter=100` was set to limit the number of iterations for the optimization algorithm.
- **Accuracy:** ~79.54%
- **F1-Score (macro avg):** ~0.80

### 4. Sample Predictions
The trained models were used to predict sentiment on new, unseen tweet samples, demonstrating their practical application and showing how each model can yield slightly different interpretations for ambiguous texts.

## Technologies Used
- **Python**
- **pandas**: For efficient data manipulation and analysis.
- **scikit-learn**: For various machine learning functionalities including TF-IDF vectorization, model training (BernoulliNB, LinearSVC, LogisticRegression), and performance evaluation metrics (`accuracy_score`, `classification_report`).

## Setup and Installation
To run this project, ensure you have the following Python libraries installed:

```bash
pip install pandas scikit-learn
```

## Usage
1. Download the dataset (e.g., `data.csv`) and place it in the project directory.
2. Run the provided Python notebook or script (`.ipynb` or `.py`).
3. The notebook will perform data loading, preprocessing, model training, evaluation, and demonstrate sample predictions.
"""
