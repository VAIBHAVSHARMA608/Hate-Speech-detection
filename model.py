from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def create_model():
    """
    Create a pipeline with TF-IDF vectorizer and Logistic Regression
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(random_state=42))
    ])
    return pipeline

def train_model(pipeline, X_train, y_train):
    """
    Train the model
    """
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(pipeline, filepath):
    """
    Save the trained model to disk
    """
    joblib.dump(pipeline, filepath)

def load_model(filepath):
    """
    Load the trained model from disk
    """
    return joblib.load(filepath)

def predict_hate_speech(model, text):
    """
    Predict if text is hate speech
    Returns 1 if hate speech, 0 otherwise
    """
    prediction = model.predict([text])
    return prediction[0]
