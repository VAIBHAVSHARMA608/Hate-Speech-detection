from data_loader import load_data, preprocess_data, split_data
from model import create_model, train_model, save_model
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load dataset (replace 'dataset.csv' with your actual file path)
    df = load_data('dataset.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Create and train model
    pipeline = create_model()
    trained_model = train_model(pipeline, X_train, y_train)

    # Evaluate model
    y_pred = trained_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    save_model(trained_model, 'hate_speech_model.pkl')
    print("Model saved as 'hate_speech_model.pkl'")

if __name__ == "__main__":
    main()
