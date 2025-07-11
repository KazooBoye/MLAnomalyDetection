# Anomaly Detection on NSL-KDD (KDD'99 variant) Dataset

This project focuses on building and evaluating various machine learning models for network intrusion detection using the NSL-KDD dataset. The primary goal is to classify network traffic into "Normal" or one of four major attack categories: Denial of Service (DoS), Probe, Remote to Local (R2L), and User to Root (U2R).

## Project Overview

The project involves several key stages:
1.  **Data Preprocessing**: Cleaning the raw KDD dataset, performing one-hot encoding for categorical features, and scaling numerical features.
2.  **Data Balancing**: Addressing the significant class imbalance in the dataset using techniques like SMOTE (Synthetic Minority Over-sampling Technique).
3.  **Model Training and Evaluation**: Implementing and evaluating the performance of multiple classification models:
    *   K-Nearest Neighbors (KNN)
    *   Random Forest (with class weighting and with SMOTE)
    *   A Hierarchical XGBoost model

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries: pandas, scikit-learn, imbalanced-learn, xgboost.

You can install the necessary packages using pip:
```bash
pip install pandas scikit-learn imbalanced-learn xgboost
```

### How to Run

Follow these steps to preprocess the data and run the models.

**Step 1: Data Preprocessing**

First, you need to process the raw `KDDTrain+.txt` and `KDDTest+.txt` files. This will create the cleaned and formatted CSV files required by the models.

```bash
# Process the training data (groups attacks into 5 classes)
python DataPreprocrss5ClassTrain.py

# Process the testing data
python DataPreprocess5ClassTest.py
```
This will generate `cleaned5Grouped_KddTrain+.csv` and `cleaned5Grouped_KddTest+.csv`.

**Step 2: Data Balancing (Optional, but Recommended)**

To handle the class imbalance, run the SMOTE script on the training data. This is required for some of the model training scripts.

```bash
# Apply SMOTE to the 5-class training data
python SmoteSupersamplingTrain.py
```
This will generate `cleaned5Grouped_KddTrain+_SMOTE.csv`.

**Step 3: Run the Models**

You can now run any of the implemented models.

**K-Nearest Neighbors (KNN)**
*Note: This script uses the SMOTE-balanced dataset (`cleaned5Grouped_KddTrain+_SMOTE.csv`). Make sure you have run Step 2.*
```bash
python KNNTrain.py
```

**Random Forest with Balanced Class Weights**
*Note: This script also uses the SMOTE-balanced dataset.*
```bash
python RandomForestClassWeights.py
```

**Hierarchical XGBoost**
*This model uses the original preprocessed training data (`cleaned5Grouped_KddTrain+.csv`) and applies its own internal resampling pipeline.*
```bash
python Hierarchical_XGBoost.py
```

## File Descriptions

- `KDDTrain+.txt`, `KDDTest+.txt`: The raw dataset files.
- `DataPreprocrss5ClassTrain.py`: Script to preprocess the training data for 5-class classification.
- `DataPreprocess5ClassTest.py`: Script to preprocess the test data for 5-class classification.
- `SmoteSupersamplingTrain.py`: Script to apply SMOTE to the preprocessed training data to create a balanced dataset.
- `KNNTrain.py`: Trains and evaluates a KNN model.
- `RandomForestClassWeights.py`: Trains and evaluates a Random Forest model using balanced class weights.
- `RandomForestSMOTETrain.py`: An alternative Random Forest implementation that uses SMOTE on the full multiclass dataset.
- `Hierarchical_XGBoost.py`: Implements a two-stage XGBoost classifier to first distinguish normal vs. attack, then classify the specific attack type.
- `*.csv`: Various data files generated during the preprocessing and balancing steps.
- `check.py`: A utility script to check for null values in the data.
