# Rainfall Prediction System

## Overview
The **Rainfall Prediction System** is a machine learning project aimed at predicting whether it will rain tomorrow based on historical weather data. The project involves data preprocessing, feature selection, model training, and evaluation.

## Project Structure

- `Rainfall_Prediction_System.ipynb`: The main Jupyter Notebook containing the code for data analysis, model building, and predictions.
- `weatherAUS.csv`: CSV file containing the dataset used for training and testing the model.

## Installation

To run this project, ensure you have the following software and libraries installed:

- Python 3.x
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

1. **Dataset**: Ensure the dataset is available in the appropriate directory or specified in the notebook.
2. **Running the Notebook**: Open the `Rainfall_Prediction_System.ipynb` file in Jupyter Notebook or Jupyter Lab.
3. **Preprocessing**: The notebook includes data cleaning, handling missing values, and feature engineering.
4. **Model Training**: Various machine learning models are trained to predict rainfall, including logistic regression, decision trees, and more.
5. **Evaluation**: Models are evaluated based on accuracy, time taken and area under ROC curve.
6. **Prediction**: After training, you can use the models to make predictions on new data.

## Model Details

This project uses multiple machine learning algorithms to find the best-performing model for rainfall prediction. The following models were explored:

- Decision Trees
- Neural Network (Multilayer Perceptron)
- Random Forest
- Light GBM
- CatBoost
- XGBoost

## Results

The performance of each model is recorded in the notebook, in terms of accuracy, time taken to train the model and area under the ROC curve. All these models are compared and best ones are reported at the end.

## Future Work

- **Hyperparameter Tuning**: Further optimization of model parameters.
- **Data Augmentation**: Incorporating additional features for better predictions.
- **Deployment**: Setting up a web interface for live predictions.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

## Acknowledgements

Special thanks to the contributors of open-source libraries and datasets that made this project possible.
