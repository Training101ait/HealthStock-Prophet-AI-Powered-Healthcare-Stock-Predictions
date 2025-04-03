# Healthcare Stocks Prediction App

This Streamlit app provides predictions for five major healthcare stocks (JNJ, PFE, MRK, ABT, UNH) using machine learning models.

## Features

- Next day's closing price prediction
- Price movement direction (up/down) prediction
- Volatility prediction
- Interactive visualizations of historical data and technical indicators
- Customizable data period selection
- Option to train new models or use existing ones
- Downloadable prediction results

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

## Usage

1. Select the data period from the sidebar (1y, 2y, 5y, 10y, or max)
2. Choose whether to train new models or use existing ones
3. Select the stocks you want to analyze
4. Click "Run Analysis" to generate predictions
5. View the results and download the predictions CSV if desired

## How It Works

The app uses historical stock data from Yahoo Finance and calculates various technical indicators (moving averages, RSI, MACD, Bollinger Bands, etc.). It then uses machine learning models to predict the next day's closing price, price movement direction, and volatility.

For each stock, the app uses the best performing model based on previous analysis:

- Price Prediction:
  - JNJ, PFE, MRK: Lasso Regression
  - ABT: Linear Regression
  - UNH: Ridge Regression

- Direction Prediction:
  - JNJ, PFE: XGBoost
  - MRK, ABT: K-Nearest Neighbors
  - UNH: Random Forest

- Volatility Prediction:
  - JNJ, UNH: Gradient Boosting
  - PFE, ABT: Random Forest
  - MRK: Linear Regression

## Important Note

This app is for educational and informational purposes only. The predictions should not be used as the sole basis for investment decisions. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.

Past performance is not indicative of future results.
