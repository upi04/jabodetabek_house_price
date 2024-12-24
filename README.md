# jabodetabek_house_price
House Price Prediction with Hyperparameter Tuning
# House Price Prediction with Hyperparameter Tuning

## Project Overview
This project focuses on predicting house prices in the Jabodetabek region using machine learning models, specifically Random Forest. The goal is to develop a model that accurately predicts house prices based on various features such as land size, building size, number of bedrooms, and other relevant factors.

### Key Features:
- **Data Collection and Cleaning**: Prepares the dataset with features like land size, building size, and location.
- **Feature Engineering**: Creates derived features such as `age_of_house` and `price_per_sqm`, and performs feature selection.
- **Hyperparameter Tuning**: Optimizes the Random Forest model using GridSearchCV and RandomizedSearchCV for the best parameters.
- **Model Evaluation**: Measures performance with metrics like Mean Squared Error (MSE) and R² Score, alongside visualizations of results.

---

## Project Structure
project/ ├── data/ # Directory for storing dataset files │ ├── raw/ # Raw dataset files │ └── processed/ # Processed dataset files ├── notebooks/ # Jupyter Notebooks for EDA and analysis │ └── eda.ipynb # Notebook for Exploratory Data Analysis ├── src/ # Source code for preprocessing and model training │ ├── feature_engineering.py # Code for feature engineering and handling missing values │ ├── model_training.py # Code for training Random Forest and hyperparameter tuning │ ├── utils.py # Utility functions for data cleaning and preprocessing │ └── init.py # Marks the directory as a Python package ├── results/ # Visualizations and performance results │ ├── figures/ # Generated figures and plots │ └── metrics.json # File to store evaluation metrics ├── requirements.txt # Required Python packages └── README.md # This file

## Steps to Run the Project

## 1. Clone the Repository
Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction
### 2. Install Dependencies
Create a virtual environment and install dependencies:

bash
Copy code
pip install -r requirements.txt
3. Data Preparation
Place the dataset in the data/raw/ directory. The dataset should include columns like price_in_rp, land_size_m2, building_size_m2, bedrooms, bathrooms, etc.

Example code to load the dataset:

python
Copy code
import pandas as pd
data = pd.read_csv('data/raw/your_dataset.csv')
print(data.head())
4. Model Training
Run the model_training.py script to train the model using Random Forest and perform hyperparameter tuning:

python
Copy code
from src.model_training import train_random_forest_model

# Train the Random Forest model
train_random_forest_model(data)
5. Evaluate Model
Model performance is evaluated using MSE and R² Score. Results and visualizations are stored in the results/ directory.

6. Visualizations
Generated visualizations include:

Residual plots
Actual vs. predicted prices
Feature importance
Hyperparameter tuning results
Results
The best Random Forest model showed the following performance:

Mean Squared Error (MSE): 100000
R² Score: 0.85
Hyperparameter tuning significantly improved the model's performance and predictive accuracy.

Future Work
Experiment with other models like Gradient Boosting and XGBoost.
Expand the dataset with more features or additional sources.
Deploy the model as a web app or API to provide predictions.
