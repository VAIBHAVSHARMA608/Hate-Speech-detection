# Hate Speech Detection Project

This project is a basic Python implementation for detecting hate speech in text data. It uses a logistic regression model with TF-IDF features to classify text as hate speech or not.

## Project Structure

- `data_loader.py`: Loads and preprocesses the dataset.
- `model.py`: Defines the hate speech detection model and prediction functions.
- `train.py`: Script to train and evaluate the model.
- `predict.py`: Script to predict hate speech on new input text.
- `requirements.txt`: Python dependencies.

## Setup

1. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# or
source venv/bin/activate      # On macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Prepare your dataset as a CSV file with columns `text` and `label` (0 for non-hate, 1 for hate).

## Usage

- To train the model:

```bash
python train.py
```

- To predict hate speech on new text:

```bash
python predict.py
```

Enter the text when prompted.

## Notes

- The dataset file path in `train.py` should be updated to your actual dataset location.
- This is a basic implementation and can be improved with more advanced models and preprocessing.
