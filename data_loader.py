import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import re

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_data(filepath):
    """
    Load dataset from CSV file.
    Assumes CSV has columns: 'text' and 'label' (0: non-hate, 1: hate)
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_text(text):
    """
    Preprocess text: lowercase, remove punctuation, tokenize, remove stopwords
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_data(df):
    """
    Apply preprocessing to the text column
    """
    df['processed_text'] = df['text'].apply(preprocess_text)
    return df

def split_data(df, test_size=0.2):
    """
    Split data into train and test sets
    """
    X = df['processed_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
