# Data Directory

This directory is intended to store the dataset for the toxicity classifier project.

## Expected Files

- `train.csv`: Training dataset containing text and labels
- `val.csv`: Validation dataset
- `test.csv`: Test dataset (optional)

## Dataset Format

Each CSV file should contain at least the following columns:
- `text`: The input text to be classified
- `label`: Binary label (0 for non-toxic, 1 for toxic)

## Getting the Data

1. Download the dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
2. Extract the downloaded files into this directory
3. Preprocess the data if necessary
4. Split into train/val/test sets

## Preprocessing

You may want to preprocess the text data before training. Common preprocessing steps include:
- Lowercasing
- Removing special characters
- Tokenization
- Removing stop words (optional)

## Note

This directory is included in .gitignore to prevent large data files from being committed to version control.
