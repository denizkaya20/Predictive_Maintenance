# Predictive Maintenance with Machine Learning

This project applies data preprocessing, feature engineering, and machine learning models to predict equipment efficiency status in a simulated manufacturing environment. The dataset contains timestamped sensor readings and categorical operational statuses.

## Project Structure

- `Predictive_Maintenance.py`: Main script for data loading, preprocessing, feature engineering, modeling, and evaluation.
- `manufacturing_6G_dataset.csv`: Source dataset (referenced in the script).

## Features

- Time-based feature extraction from timestamp
- Outlier detection and capping using IQR
- Label encoding for categorical variables
- Feature scaling and interaction terms
- Model training using:
  - PyCaret (with automatic model comparison and tuning)
  - Logistic Regression (manual baseline)
- Evaluation with confusion matrix, classification report, ROC-AUC, and feature importance

## Models Used

- **PyCaret AutoML Framework**
- **Logistic Regression (manual implementation)**

## Highlights

- Custom feature engineering including:
  - `Power_Efficiency_Ratio`
  - `Thermal_Mechanical_Stress`
  - `Quality_Speed_Interaction`
  - `Network_Performance`
- PyCaret pipeline with:
  - Cross-validation using time-series folds
  - Multicollinearity reduction
  - Automated hyperparameter tuning
- Visualization of model performance and comparison

## Results

The project compares PyCaret's best model against a manually tuned logistic regression model. Accuracy, confusion matrix, and AUC are used for evaluation. Feature importance is visualized for interpretability.

## Requirements

Install required packages with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pycaret
```

## How to Run

Make sure you have the dataset `manufacturing_6G_dataset.csv` in the same directory. Then run:

```bash
python Predictive_Maintenance.py
```

## Notes

- The dataset may be synthetic or simulated.
- Some models showed near-perfect accuracy, suggesting potential overfitting or synthetic characteristics.
- Logistic Regression provides a more interpretable and conservative benchmark.

---

### 📬 Author

Deniz Kaya – 2025
