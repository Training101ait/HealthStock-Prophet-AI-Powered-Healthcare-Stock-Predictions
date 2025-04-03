# Healthcare Stocks Prediction System

This project implements a machine learning system to predict the following for five major healthcare stocks (JNJ, PFE, MRK, ABT, UNH):
1. Next day's closing price
2. Price movement direction (up/down)
3. Volatility

## Project Structure

```
stock_prediction/
├── data/                      # Stock historical data
├── models/                    # Trained machine learning models
│   ├── price_prediction/      # Models for price prediction
│   ├── direction_prediction/  # Models for direction prediction
│   └── volatility_prediction/ # Models for volatility prediction
├── notebooks/                 # Jupyter/Colab notebooks
├── app/                       # Streamlit app for live predictions
├── validation_results/        # Validation and backtesting results
├── utils.py                   # Utility functions
├── predict_stocks.py          # Standalone prediction script
├── validate_models.py         # Model validation script
└── README.md                  # This file
```

## Components

### 1. Google Colab Notebook

The `notebooks/healthcare_stocks_prediction.ipynb` file contains a comprehensive notebook that can be run in Google Colab. It includes:
- Data fetching from Yahoo Finance
- Technical indicator calculation
- Model training and evaluation
- Visualization of stock data and predictions
- Next-day predictions

### 2. Standalone Python Scripts

The standalone Python scripts allow you to run the prediction system on your laptop:

- `utils.py`: Contains utility functions for data processing and visualization
- `predict_stocks.py`: Main script for fetching data, training models, and making predictions

To run the standalone script:
```bash
python predict_stocks.py                   # Process all stocks
python predict_stocks.py --symbol JNJ      # Process a specific stock
python predict_stocks.py --no-train        # Use existing models
python predict_stocks.py --no-visualize    # Skip visualizations
```

### 3. Streamlit App

The Streamlit app provides an interactive web interface for making live predictions:

- `app/app.py`: Main Streamlit application
- `app/utils.py`: Utility functions for the app
- `app/requirements.txt`: Dependencies for the app

To run the Streamlit app:
```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd stock_prediction
```

2. Install dependencies for standalone scripts:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost yfinance plotly joblib
```

3. For the Streamlit app:
```bash
cd app
pip install -r requirements.txt
```

## Usage

### Google Colab Notebook

1. Upload the `notebooks/healthcare_stocks_prediction.ipynb` file to Google Colab
2. Run all cells to fetch data, train models, and make predictions

### Standalone Scripts

1. Run the prediction script:
```bash
python predict_stocks.py
```

2. Check the generated predictions:
```bash
cat predictions_YYYYMMDD.csv
```

3. Validate model accuracy:
```bash
python validate_models.py
```

### Streamlit App

1. Start the Streamlit app:
```bash
cd app
streamlit run app.py
```

2. Use the sidebar controls to:
   - Select data period
   - Choose whether to train new models
   - Select stocks to analyze
   - Run the analysis

## Model Performance

Based on backtesting results, our models show strong predictive performance:

### Direction Prediction Accuracy
- JNJ: 84.75% (highest accuracy)
- PFE: 79.66%
- MRK: 67.80% (lowest accuracy)
- ABT: 77.97%
- UNH: 79.66%

### Price Prediction (R² Score)
- JNJ: 0.939
- PFE: 0.822
- MRK: 0.887
- ABT: 0.928
- UNH: 0.976 (highest accuracy)

### Volatility Prediction (R² Score)
- JNJ: 0.750
- PFE: 0.975 (highest accuracy)
- MRK: 0.783
- ABT: 0.718
- UNH: 0.780

## Important Notes

- This system is for educational and informational purposes only
- Past performance is not indicative of future results
- Always conduct your own research before making investment decisions
- The predictions should be used as one of many tools for investment analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.
