import os
import joblib
from django.conf import settings
from ml_model.train import clean_and_preprocess_text

# Paths to the saved ML artifacts
VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'saved_models', 'tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_model', 'saved_models', 'spam_detector_model.joblib')

# Load the model and vectorizer at startup
# This is a production best practice: loading them once in memory rather than reading from disk on every request.
try:
    if os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
        model = joblib.load(MODEL_PATH)
        print("[OK] Machine Learning model and vectorizer loaded successfully!")
    else:
        vectorizer = None
        model = None
        print("[WARNING] Machine Learning artifacts not found. Please run the training script first!")
except Exception as e:
    vectorizer = None
    model = None
    print(f"[ERROR] Error loading Machine Learning artifacts: {e}")


def predict_email(email_text):
    """
    Takes raw email text, preprocesses it, runs it through the TF-IDF vectorizer,
    and returns the prediction label (Spam/Ham) and confidence score.
    """
    if not model or not vectorizer:
        return {
            'prediction_label': 'Error',
            'confidence': 0.0,
            'cleaned_text': '',
            'error': 'Machine learning model is not loaded. Please run the model training script.'
        }
        
    if not email_text.strip():
        return {
            'prediction_label': 'Error',
            'confidence': 0.0,
            'cleaned_text': '',
            'error': 'Email text cannot be empty.'
        }

    # 1. Clean and preprocess the input email text
    cleaned_text = clean_and_preprocess_text(email_text)
    
    # 2. Transform the text using the loaded TF-IDF Vectorizer
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # 3. Predict the label (0 = Ham, 1 = Spam)
    prediction = model.predict(vectorized_text)[0]
    
    # 4. Get the probabilities
    # predict_proba returns an array: [[probability_of_class_0, probability_of_class_1]]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # 5. Extract label and confidence level
    if prediction == 1:
        label = 'Spam'
        confidence = probabilities[1] * 100 # Probability of class 1 (Spam)
    else:
        label = 'Ham'
        confidence = probabilities[0] * 100 # Probability of class 0 (Ham)
        
    return {
        'prediction_label': label,
        'confidence': round(confidence, 2),
        'cleaned_text': cleaned_text,
        'error': None
    }
