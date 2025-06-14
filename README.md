# **Toxicity & Threat Classifier for Online Communities**

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch Version">
  <img src="https://img.shields.io/badge/🤗%20Transformers-4.x-yellow.svg" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</div>

---

## **Table of Contents**

* [Project Overview](#project-overview)
* [Key Features](#key-features)
* [Model Performance](#model-performance)
* [Tech Stack](#tech-stack)
* [Repository Structure](#repository-structure)
* [Setup and Installation](#setup-and-installation)
* [How to Use](#how-to-use)
* [Challenges & Key Learnings](#challenges--key-learnings)
* [License](#license)
* [Contact](#contact)

---

## **Project Overview**

This project addresses the critical need for effective content moderation on online platforms. Simple keyword filters are often insufficient for capturing the nuance of harmful language. This repository contains the complete workflow for building, training, and evaluating a sophisticated, context-aware model capable of performing **multi-label text classification** on online comments.

The model is fine-tuned on the classic "Jigsaw Toxic Comment Classification Challenge" dataset and can identify whether a comment falls into one or more of six categories: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`.

The final model achieves a **F1-score of 0.83** for the primary "toxic" label after a robust optimization process, demonstrating a strong balance between precision and recall.

## **Key Features**

* **Multi-Label Classification:** A single comment can be correctly classified with multiple toxic labels (e.g., both `toxic` and `insult`).
* **Context-Aware:** Built on the **DistilBERT** transformer architecture, allowing it to understand context far better than traditional methods.
* **Handles Class Imbalance:** Implements a custom weighted loss function (`BCEWithLogitsLoss` with `pos_weight`) to effectively train on a highly imbalanced dataset.
* **Optimized for F1-Score:** Includes a threshold optimization step to find the ideal decision boundary for each label, maximizing the balance between precision and recall.
* **Reproducible Workflow:** The entire process, from data preparation to training and evaluation, is documented in a clean Jupyter/Colab notebook (`analysis_and_modeling.ipynb`).

## **Model Performance**

A key challenge was moving from a model with high recall but low precision to one that was well-balanced. This was achieved by optimizing the classification threshold for each label individually.

Below is the final performance on the held-out test set using the optimized thresholds:

| Label | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **toxic** | **0.82** | **0.83** | **0.83** |
| severe\_toxic | 0.44 | 0.71 | 0.54 |
| obscene | 0.82 | 0.88 | 0.85 |
| threat | 0.51 | 0.70 | 0.59 |
| insult | 0.71 | 0.84 | 0.77 |
| identity\_hate| 0.48 | 0.63 | 0.54 |
| **---** | **---** | **---** | **---** |
| **Weighted Avg**| **0.76** | **0.83** | **0.79** |

### Model Performance Visualizations

#### Confusion Matrices for Each Label
![Confusion Matrices](CONFUSION%20MATRICES-1.png)

#### Distribution of Toxicity Labels
![Distribution of Toxicity](Distribution%20of%20toxicity-1.png)

## **Tech Stack**

* **Core Libraries:** Python, PyTorch, Hugging Face (Transformers, Datasets, Evaluate)
* **Data Handling:** Pandas, NumPy (`<2.0` for compatibility)
* **Evaluation & Metrics:** Scikit-learn
* **Development Environment:** Google Colab, Jupyter Notebook

## **Repository Structure**

```
├── .gitignore
├── README.md                # You are here!
├── requirements.txt         # Project dependencies
├── notebooks/
│   └── analysis_and_modeling.ipynb  # Main notebook with the complete workflow
└── src/
    └── distilbert-toxicity-classifier/  # Saved pre-trained model and tokenizer files
```

## **Setup and Installation**

To set up this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)Jupiter-Plantagenet/toxicity-classifier.git
    cd toxicity-classifier
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Kaggle API Key (Optional, for re-training):** To re-run the data download from the notebook, place your `kaggle.json` API key in the root directory.

## **How to Use**

There are two ways to use this project:

#### 1. Explore the Full Process
Open the `notebooks/analysis_and_modeling.ipynb` file in Jupyter or Google Colab to see the entire process, from data loading and cleaning to model training and evaluation.

#### 2. Use the Pre-trained Model for Inference
You can easily load the final trained model from the `src/` directory to classify new text.

```python
from transformers import pipeline

# Load the model from the local directory
model_path = "./src/distilbert-toxicity-classifier/"
classifier = pipeline("text-classification", model=model_path, tokenizer=model_path, return_all_scores=True)

# Classify a new comment
comment = "You are an amazing and talented person! What a great idea."
results = classifier(comment)

# Print the results
import pandas as pd
df_results = pd.DataFrame(results[0]).sort_values(by='score', ascending=False)
print(df_results)
```

## **Challenges & Key Learnings**

This project involved navigating several real-world software development challenges, particularly around dependency management:

* **NumPy 2.0 Incompatibility:** The project was developed during the major NumPy 2.0 release. This caused a `ValueError: numpy.dtype size changed...` due to a binary incompatibility with the `datasets` library. **Solution:** Pinned the dependency to `numpy<2.0` to ensure a stable environment.
* **Hugging Face API Evolution:** The `transformers` library is constantly evolving. Debugging involved updating deprecated arguments (like `evaluation_strategy` to `eval_strategy`) and adapting custom class methods (`compute_loss`) to handle new, unexpected arguments passed by the `Trainer`.

These challenges highlight the importance of careful environment management and writing flexible code that can adapt to library updates.

## **License**

This project is licensed under the MIT License.

## **Contact**

Created by **[George Akor]** - Feel free to connect!

* **GitHub:** `https://github.com/Jupiter-Plantagenet`
* **LinkedIn:** `www.linkedin.com/in/george-akor-65953819a`