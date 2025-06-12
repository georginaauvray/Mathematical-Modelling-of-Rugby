
# Rugby Peformance Modelling - Expected Threat and Penalty Decision Models

This project contains statistical models developed as part of a Master's thesis in Applied Mathematics and Statistics at Uppsala University. The models apply machine learning to rugby union, adapting methods commonly used in football analytics.

## Contents

- `Expected Threat.py`: Builds a phase-based expected threat model for rugby using logistic regression.
- `Penalty.py`: Predicts optimal decisions at penalties using logistic regression, linear regression, and gradient boosting.
- `Plotting pitches.py` Provides visualisation tools for plotting rugby pitches.

## Project Goals

1. Develop an expected threat model tailored to rugby's unique structure.
2. Model and evaluate decisions made at penalties based on pitch position, match context, and success probabilities.

## Requirements

This project uses Python 3 and requires the following packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

To install dependencies:
```bash
pip install -r requirements.txt
```

## How to Use
## 1. Expected Threat Model
Run `Expected Threat.py` to:
- Preprocess rugby event data
- Extract features from possession phases
- Train a logistic regression model to predict scoring likelihood

## 2. Penalty Decision Model
Run `Penalty.py` to: 
- Fit sub-models for goal kick and kick-to-touch success
- Evaluate decision probabilities using gradient boosting
- Simulate or analyse historical decisions

## 3. Pitch Visualisations
Use `Plotting pitches.py` to generate custom pitch diagrams.

## Author
Georgina Auvray \
Master's Thesis in Applied Mathematics and Statistics \
Uppsala University, 2025 

With support from Professor David Sumpter and the Irish Rugby Football union

## Licence
This project is licensed for academic and non-commerical use. Please contact the author for reuse or collaboration.



