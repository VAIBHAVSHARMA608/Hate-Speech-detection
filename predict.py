from model import load_model, predict_hate_speech
from data_loader import preprocess_text

def main():
    # Load trained model
    model = load_model('hate_speech_model.pkl')

    # Example text to predict
    text = input("Enter text to check for hate speech: ")
    processed_text = preprocess_text(text)
    prediction = predict_hate_speech(model, processed_text)

    if prediction == 1:
        print("This text is classified as hate speech.")
    else:
        print("This text is not classified as hate speech.")

if __name__ == "__main__":
    main()
