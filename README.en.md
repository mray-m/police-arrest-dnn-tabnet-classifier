# Police Arrest Prediction — DNN & TabNet

This project is a binary classification study on police arrest records, aiming to predict whether an arrest is an **"on-view arrest"** (made on the spot during a patrol) or not. A **Deep Neural Network (DNN)** and a **TabNet Classifier** were trained and compared.

## Contents

| File | Description |
|---|---|
| `dataset_linki.txt` | Link to download the raw dataset |
| `preprocessing.ipynb` | Data cleaning and preprocessing steps |
| `DNNClassifier.ipynb` | Deep Neural Network model built with PyTorch |
| `TabNetClassifier.ipynb` | TabNet model built with pytorch-tabnet |
| `TheRookies_Sunum.pdf` | Project presentation |

## About the Project

The dataset consists of real police arrest records, where each record contains various features of an arrest event (such as location, date, and time). The goal of the project is to predict, based on these features, whether an arrest was made on the spot during a routine patrol ("on-view arrest") or as a result of prior planning or investigation. This is a meaningful classification problem for analyzing public safety data transparency and police operations.

The raw data is cleaned and prepared for modeling in `preprocessing.ipynb`; the same processed dataset is then used to train two different deep learning approaches — a classic DNN and TabNet, an architecture specifically designed for tabular data — and their performances are compared.

## Methodology

- **Target variable**: `is_onview_arrest`
- **Preprocessing**: Categorical variables were encoded using One-Hot Encoding for the DNN and Label Encoding for TabNet. Low-variance features were removed with `VarianceThreshold`, and for the DNN, the top 1000 most informative features were further selected using `SelectKBest` (ANOVA F-test).
- **Class imbalance**: Since the positive class was underrepresented in the training set, a weighted `BCEWithLogitsLoss` with `pos_weight` was used for the DNN.
- **Model selection**: Hyperparameter optimization for the DNN was performed via random search (hidden_dim, dropout, learning rate).

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| DNN Classifier | 0.795 | 0.570 | 0.607 | 0.588 | 0.833 |
| TabNet Classifier | 0.842 | 0.719 | 0.569 | 0.635 | 0.877 |

TabNet achieved stronger overall performance with higher accuracy, precision, and ROC-AUC, while the DNN performed slightly better in terms of recall.

## Technologies Used

- **Python**
- **Pandas / NumPy** — data processing
- **Scikit-learn** — preprocessing, feature selection, model evaluation
- **PyTorch** — building and training the DNN model
- **pytorch-tabnet** — building and training the TabNet model
- **Jupyter Notebook** — development environment
