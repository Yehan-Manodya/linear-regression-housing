# Linear Regression – Housing Price Prediction

## Project Overview
This project implements a Linear Regression machine learning model to predict house prices using a real-world housing dataset.  
The complete machine learning workflow is covered, from data analysis to model evaluation.

---

## Dataset
- California Housing Dataset (CSV format)
- Contains numerical and categorical features
- Target variable: `median_house_value`

---

## Project Structure
Linear-regression/
│
├── data/
│ └── housing.csv
│
├── notebooks/
│ └── eda.ipynb
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ └── evaluate.py
│
├── venv/
├── README.md
├── requirements.txt
└── .gitignore


---

## Exploratory Data Analysis (EDA)
- Dataset inspection and summary statistics
- Identification of missing values
- Correlation analysis using heatmaps
- Analysis of categorical feature (`ocean_proximity`)

EDA is performed in `notebooks/eda.ipynb`.

---

## Data Preprocessing
The following preprocessing steps were applied:
- Missing values filled using column mean
- Categorical feature encoded using one-hot encoding
- Feature scaling using StandardScaler
- Train-test split (80% training, 20% testing)

---

## Model Training
- Linear Regression model from `scikit-learn`
- Model trained using preprocessed training data

---

## Model Evaluation
Evaluation metrics used:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

A scatter plot of **Actual vs Predicted house prices** is generated to visualize model performance.

---

## How to Run the Project

1. Activate virtual environment
```bash
venv\Scripts\activate
python src/evaluate.py
