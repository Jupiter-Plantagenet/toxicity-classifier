# Toxicity Classifier

A machine learning model for detecting toxic content in text using DistilBERT.

## Project Structure

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── data/                   # Placeholder for Kaggle dataset
├── notebooks/              # Jupyter notebooks for analysis and modeling
│   └── analysis_and_modeling.ipynb
├── src/                    # Source code
│   ├── __init__.py
│   ├── model.py           # Model architecture and utilities
│   ├── train.py           # Training script
│   └── utils.py           # Helper functions
└── distilbert-toxicity-classifier.safetensors  # Pre-trained model weights
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd toxicity-classifier
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset in the `data/` directory
2. Run the Jupyter notebooks for analysis and modeling:
   ```bash
   jupyter notebook notebooks/analysis_and_modeling.ipynb
   ```
3. Or use the training script:
   ```bash
   python src/train.py
   ```

## Model

The model uses DistilBERT for toxicity classification, providing a good balance between performance and efficiency.

## License

MIT
