# Cape Town Crime Hotspot Prediction

This repository contains the code and deployment pipeline for my BSc Honours Data Science dissertation: "Predicting Spatiotemporal Crime Hotspots Using Machine Learning in Cape Town, South Africa."

## Live Dashboard
You can view the interactive Streamlit dashboard here: 
[https://crime-hotspot-prediction-nomz3ddcxdscx8b8dzcrqj.streamlit.app/](https://crime-hotspot-prediction-nomz3ddcxdscx8b8dzcrqj.streamlit.app/)

## Project Overview
This project applies three machine learning algorithms (Logistic Regression, Random Forest, and XGBoost) to predict crime hotspots at the SAPS station level in Cape Town. 

*   Dataset: SAPS Annual Crime Records (2005-2024) from the DataFirst UCT repository.
*   Features: 159 engineered features (lags, rolling means, trends, and geographic coordinates).
*   Evaluation: Models were trained on 2008-2019 data and tested on 2020-2024 data.
*   Best Model: Random Forest achieved a mean ROC-AUC of 0.9318.

## Repository Structure
*   app/main.py: The Streamlit dashboard code.
*   src/: Core Python modules (config, data preprocessor, model trainer, hotspot predictor).
*   data/processed/: The final engineered dataset (features.parquet) and forecasts.
*   models/: The trained .pkl models and evaluation metrics (metrics.csv).
*   outputs/: EDA charts and feature importance figures.
*   train_models.py: The script to train and evaluate the models.
*   requirements.txt: Python dependencies for deployment.

## How to Run Locally
1. Clone this repository.
2. Install dependencies: pip install -r requirements.txt
3. Run the dashboard: streamlit run app/main.py

## Author
Mitchley Tapiwa Dinner
BSc Honours in Data Science
Eduvos