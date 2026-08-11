import os
import urllib.request
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading missing NLTK resources...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Define Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'saved_models')

# Create directories if they do not exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, 'spam_dataset.tsv')
# Public repository hosting the standard SMS/Email Spam Collection dataset (TSV format)
DATASET_URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"


def download_dataset():
    """
    Downloads the public spam dataset if it's not already available locally.
    This ensures the project is self-contained and run-ready on any system.
    """
    if not os.path.exists(DATASET_FILE):
        print(f"Downloading dataset from: {DATASET_URL}")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_FILE)
            print("Dataset downloaded successfully!")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Please ensure you are connected to the internet, or place a tab-separated spam file at:", DATASET_FILE)
            raise e
    else:
        print("Dataset found locally, skipping download.")


def clean_and_preprocess_text(text):
    """
    Educational Preprocessing Pipeline:
    1. Lowecases the text (so 'Spam' and 'spam' are treated the same).
    2. Removes special characters, numbers, and punctuation.
    3. Tokenizes the text (breaks sentences down into individual words).
    4. Removes Stop Words (common words like 'the', 'is', 'at' that add no classification value).
    5. Lemmatizes the words (reduces words to their base dictionary form, e.g., 'running' -> 'run').
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercasing
    text = text.lower()
    
    # 2. Remove special characters/punctuation & numbers (keep only letters)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization (splitting text into words)
    words = word_tokenize(text)
    
    # 4. Remove Stop Words & Lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Process each word: only keep if not a stopword, and reduce to lemma
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    # Rejoin words back into a single string space-separated
    return " ".join(cleaned_words)


def train_models():
    # Step 1: Get the dataset
    download_dataset()
    
    # Step 2: Load the dataset (TSV - Tab Separated Values)
    # The dataset has two columns: 'label' (ham or spam) and 'message' (the email/SMS text)
    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE, sep='\t', names=['label', 'message'])
    print(f"Dataset loaded. Total records: {len(df)}")
    print(df['label'].value_counts())
    
    # Step 3: Clean and preprocess the messages
    print("Preprocessing text (this might take a few seconds)...")
    df['cleaned_message'] = df['message'].apply(clean_and_preprocess_text)
    
    # Drop rows that ended up completely empty after preprocessing (if any)
    df = df[df['cleaned_message'] != '']
    
    # Map text labels to numbers: ham -> 0, spam -> 1
    # Machine Learning algorithms require numeric inputs, not text categories.
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Step 4: Split data into features (X) and target labels (y)
    X = df['cleaned_message']
    y = df['label_num']
    
    # Split into 80% Training set and 20% Testing set
    # random_state=42 is used to ensure reproducible splits across runs
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Step 5: Feature Extraction - TF-IDF Vectorizer
    # TF-IDF (Term Frequency-Inverse Document Frequency) measures how important a word is in a document
    # relative to the entire dataset, converting words into a numerical representation.
    print("Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000) # Keep top 5000 most important terms
    
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    # Step 6: Model Training & Evaluation
    # 6.1: Multinomial Naive Bayes (Fast, works great with term counts/frequencies)
    print("\nTraining Multinomial Naive Bayes model...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vectorized, y_train)
    
    nb_preds = nb_model.predict(X_test_vectorized)
    nb_accuracy = accuracy_score(y_test, nb_preds)
    print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, nb_preds, target_names=['Ham', 'Spam']))
    
    # 6.2: Logistic Regression (Estimates probability of a binary classification problem)
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vectorized, y_train)
    
    lr_preds = lr_model.predict(X_test_vectorized)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_preds, target_names=['Ham', 'Spam']))
    
    # Step 7: Compare and Save the Best Model
    if nb_accuracy >= lr_accuracy:
        best_model = nb_model
        best_model_name = "Naive Bayes"
        best_accuracy = nb_accuracy
    else:
        best_model = lr_model
        best_model_name = "Logistic Regression"
        best_accuracy = lr_accuracy
        
    print(f"\nWinner: {best_model_name} with {best_accuracy * 100:.2f}% accuracy!")
    
    # Save the TF-IDF Vectorizer
    vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved to: {vectorizer_path}")
    
    # Save the trained model
    model_path = os.path.join(MODEL_DIR, 'spam_detector_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Trained model saved to: {model_path}")
    
    # Save a metadata text file to record the results of the training
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"Best Model Name: {best_model_name}\n")
        f.write(f"Best Model Accuracy: {best_accuracy * 100:.2f}%\n")
        f.write(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%\n")
        f.write(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%\n")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == '__main__':
    train_models()
