import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, confusion_matrix, classification_report
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    calculate_technical_indicators, prepare_data, 
    STOCK_DESCRIPTIONS, PRICE_FEATURES, DIRECTION_FEATURES, VOLATILITY_FEATURES
)

def load_historical_data(symbol):
    """
    Load historical data from CSV file
    
    Parameters:
    symbol (str): Stock symbol
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    try:
        df = pd.read_csv(f'data/{symbol}_historical.csv', index_col='Date', parse_dates=True)
        print(f"Loaded data from data/{symbol}_historical.csv")
        return df
    except FileNotFoundError:
        print(f"Could not find data file for {symbol}")
        return None

def load_models(symbol):
    """
    Load trained models for a stock
    
    Parameters:
    symbol (str): Stock symbol
    
    Returns:
    dict: Dictionary containing models and preprocessing objects
    """
    try:
        # Load price prediction model
        price_model = joblib.load(f'models/price_prediction/{symbol}_price_model.pkl')
        price_scaler = joblib.load(f'models/price_prediction/{symbol}_price_scaler.pkl')
        price_imputer = joblib.load(f'models/price_prediction/{symbol}_price_imputer.pkl')
        
        # Load direction prediction model
        direction_model = joblib.load(f'models/direction_prediction/{symbol}_direction_model.pkl')
        direction_scaler = joblib.load(f'models/direction_prediction/{symbol}_direction_scaler.pkl')
        direction_imputer = joblib.load(f'models/direction_prediction/{symbol}_direction_imputer.pkl')
        
        # Load volatility prediction model
        volatility_model = joblib.load(f'models/volatility_prediction/{symbol}_volatility_model.pkl')
        volatility_scaler = joblib.load(f'models/volatility_prediction/{symbol}_volatility_scaler.pkl')
        volatility_imputer = joblib.load(f'models/volatility_prediction/{symbol}_volatility_imputer.pkl')
        
        return {
            'price': {
                'model': price_model,
                'scaler': price_scaler,
                'imputer': price_imputer
            },
            'direction': {
                'model': direction_model,
                'scaler': direction_scaler,
                'imputer': direction_imputer
            },
            'volatility': {
                'model': volatility_model,
                'scaler': volatility_scaler,
                'imputer': volatility_imputer
            }
        }
    except FileNotFoundError as e:
        print(f"Could not load models for {symbol}: {e}")
        return None

def perform_backtesting(symbol, test_period=60):
    """
    Perform backtesting on historical data
    
    Parameters:
    symbol (str): Stock symbol
    test_period (int): Number of days to use for testing
    
    Returns:
    dict: Dictionary containing backtesting results
    """
    print(f"\n{'='*50}\nPerforming backtesting for {symbol}\n{'='*50}")
    
    # Load historical data
    df = load_historical_data(symbol)
    if df is None:
        return None
    
    # Calculate technical indicators
    df_processed = calculate_technical_indicators(df)
    
    # Load models
    models = load_models(symbol)
    if models is None:
        return None
    
    # Get the test data (last test_period days)
    test_data = df_processed.iloc[-test_period:]
    
    # Price prediction backtesting
    price_predictions = []
    price_actuals = []
    
    # Direction prediction backtesting
    direction_predictions = []
    direction_actuals = []
    
    # Volatility prediction backtesting
    volatility_predictions = []
    volatility_actuals = []
    
    # Iterate through each day in the test period
    for i in range(len(test_data) - 1):  # -1 because we need the next day's actual values
        # Get the current day's data
        current_data = test_data.iloc[i:i+1][PRICE_FEATURES]
        
        # Get the next day's actual values
        next_day_price = test_data.iloc[i+1]['Close']
        next_day_direction = 1 if test_data.iloc[i+1]['Close'] > test_data.iloc[i]['Close'] else 0
        next_day_volatility = test_data.iloc[i+1]['Volatility']
        
        # Price prediction
        current_data_imputed = models['price']['imputer'].transform(current_data)
        current_data_scaled = models['price']['scaler'].transform(current_data_imputed)
        price_pred = models['price']['model'].predict(current_data_scaled)[0]
        
        # Direction prediction
        current_data_imputed = models['direction']['imputer'].transform(current_data)
        current_data_scaled = models['direction']['scaler'].transform(current_data_imputed)
        direction_pred = models['direction']['model'].predict(current_data_scaled)[0]
        
        # Volatility prediction
        current_data_imputed = models['volatility']['imputer'].transform(current_data)
        current_data_scaled = models['volatility']['scaler'].transform(current_data_imputed)
        volatility_pred = models['volatility']['model'].predict(current_data_scaled)[0]
        
        # Store predictions and actuals
        price_predictions.append(price_pred)
        price_actuals.append(next_day_price)
        
        direction_predictions.append(direction_pred)
        direction_actuals.append(next_day_direction)
        
        volatility_predictions.append(volatility_pred)
        volatility_actuals.append(next_day_volatility)
    
    # Calculate metrics for price prediction
    price_mse = mean_squared_error(price_actuals, price_predictions)
    price_mae = mean_absolute_error(price_actuals, price_predictions)
    price_r2 = r2_score(price_actuals, price_predictions)
    
    # Calculate metrics for direction prediction
    direction_accuracy = accuracy_score(direction_actuals, direction_predictions)
    direction_cm = confusion_matrix(direction_actuals, direction_predictions)
    direction_report = classification_report(direction_actuals, direction_predictions, output_dict=True)
    
    # Calculate metrics for volatility prediction
    volatility_mse = mean_squared_error(volatility_actuals, volatility_predictions)
    volatility_mae = mean_absolute_error(volatility_actuals, volatility_predictions)
    volatility_r2 = r2_score(volatility_actuals, volatility_predictions)
    
    # Print results
    print(f"\nPrice Prediction Results for {symbol}:")
    print(f"MSE: {price_mse:.4f}")
    print(f"MAE: {price_mae:.4f}")
    print(f"R2: {price_r2:.4f}")
    
    print(f"\nDirection Prediction Results for {symbol}:")
    print(f"Accuracy: {direction_accuracy:.4f}")
    print("Confusion Matrix:")
    print(direction_cm)
    print("Classification Report:")
    print(classification_report(direction_actuals, direction_predictions))
    
    print(f"\nVolatility Prediction Results for {symbol}:")
    print(f"MSE: {volatility_mse:.4f}")
    print(f"MAE: {volatility_mae:.4f}")
    print(f"R2: {volatility_r2:.4f}")
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(14, 7))
    plt.plot(price_actuals, label='Actual', color='blue')
    plt.plot(price_predictions, label='Predicted', color='red')
    plt.title(f'{symbol} - Actual vs Predicted Prices (Backtesting)')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'backtesting_{symbol}_price.png')
    plt.close()
    
    # Plot actual vs predicted volatility
    plt.figure(figsize=(14, 7))
    plt.plot(volatility_actuals, label='Actual', color='blue')
    plt.plot(volatility_predictions, label='Predicted', color='red')
    plt.title(f'{symbol} - Actual vs Predicted Volatility (Backtesting)')
    plt.xlabel('Days')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'backtesting_{symbol}_volatility.png')
    plt.close()
    
    # Return results
    return {
        'symbol': symbol,
        'price': {
            'mse': price_mse,
            'mae': price_mae,
            'r2': price_r2,
            'actuals': price_actuals,
            'predictions': price_predictions
        },
        'direction': {
            'accuracy': direction_accuracy,
            'confusion_matrix': direction_cm,
            'report': direction_report,
            'actuals': direction_actuals,
            'predictions': direction_predictions
        },
        'volatility': {
            'mse': volatility_mse,
            'mae': volatility_mae,
            'r2': volatility_r2,
            'actuals': volatility_actuals,
            'predictions': volatility_predictions
        }
    }

def main():
    """Main function to test and validate prediction accuracy"""
    # Create results directory
    os.makedirs('validation_results', exist_ok=True)
    
    # Stocks to test
    stocks = list(STOCK_DESCRIPTIONS.keys())
    
    # Perform backtesting for each stock
    all_results = {}
    
    for symbol in stocks:
        results = perform_backtesting(symbol)
        if results:
            all_results[symbol] = results
    
    # Create summary DataFrame
    summary = {
        'Symbol': [],
        'Price MSE': [],
        'Price MAE': [],
        'Price R2': [],
        'Direction Accuracy': [],
        'Direction Precision': [],
        'Direction Recall': [],
        'Direction F1': [],
        'Volatility MSE': [],
        'Volatility MAE': [],
        'Volatility R2': []
    }
    
    for symbol, results in all_results.items():
        summary['Symbol'].append(symbol)
        summary['Price MSE'].append(results['price']['mse'])
        summary['Price MAE'].append(results['price']['mae'])
        summary['Price R2'].append(results['price']['r2'])
        summary['Direction Accuracy'].append(results['direction']['accuracy'])
        summary['Direction Precision'].append(results['direction']['report']['weighted avg']['precision'])
        summary['Direction Recall'].append(results['direction']['report']['weighted avg']['recall'])
        summary['Direction F1'].append(results['direction']['report']['weighted avg']['f1-score'])
        summary['Volatility MSE'].append(results['volatility']['mse'])
        summary['Volatility MAE'].append(results['volatility']['mae'])
        summary['Volatility R2'].append(results['volatility']['r2'])
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('validation_results/backtesting_summary.csv', index=False)
    
    print("\nBacktesting Summary:")
    print(summary_df)
    
    # Plot summary results
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df['Symbol'], summary_df['Direction Accuracy'])
    plt.title('Direction Prediction Accuracy by Stock')
    plt.xlabel('Stock')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(summary_df['Direction Accuracy']):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig('validation_results/direction_accuracy_summary.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.bar(summary_df['Symbol'], summary_df['Price R2'])
    plt.title('Price Prediction R² Score by Stock')
    plt.xlabel('Stock')
    plt.ylabel('R² Score')
    for i, v in enumerate(summary_df['Price R2']):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig('validation_results/price_r2_summary.png')
    plt.close()
    
    print("\nValidation complete. Results saved to validation_results directory.")

if __name__ == "__main__":
    main()
