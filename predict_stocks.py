import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
import joblib
import os
import datetime
import argparse
from utils import (
    calculate_technical_indicators, prepare_data, 
    visualize_stock_price_history, visualize_technical_indicators,
    visualize_price_predictions, visualize_direction_predictions, 
    visualize_volatility_predictions, create_output_dirs,
    STOCK_DESCRIPTIONS, PRICE_FEATURES, DIRECTION_FEATURES, VOLATILITY_FEATURES
)

def fetch_stock_data(symbol, period='5y', save=True):
    """
    Fetch historical stock data using Yahoo Finance API
    
    Parameters:
    symbol (str): Stock symbol
    period (str): Time period to fetch data for (default: '5y' for 5 years)
    save (bool): Whether to save the data to a CSV file
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    print(f"Fetching data for {symbol}...")
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Convert Date column to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index again
    df = df.set_index('Date')
    
    print(f"Fetched {len(df)} rows of data for {symbol}")
    
    if save:
        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/{symbol}_historical.csv')
        print(f"Data saved to data/{symbol}_historical.csv")
    
    return df

def train_price_model(df, symbol, save=True):
    """
    Train model for next day closing price prediction
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    save (bool): Whether to save the model
    
    Returns:
    tuple: model, scaler, imputer, mse, mae, r2, y_test, y_pred
    """
    print(f"Training price prediction model for {symbol}...")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, imputer = prepare_data(
        df, PRICE_FEATURES, 'Next_Day_Close', 'regression'
    )
    
    # Select the best model based on our previous analysis
    if symbol in ['JNJ', 'PFE', 'MRK']:
        model = Lasso()
    elif symbol == 'ABT':
        model = LinearRegression()
    else:  # UNH
        model = Ridge()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    if save:
        # Save the model and preprocessing objects
        os.makedirs('models/price_prediction', exist_ok=True)
        joblib.dump(model, f'models/price_prediction/{symbol}_price_model.pkl')
        joblib.dump(scaler, f'models/price_prediction/{symbol}_price_scaler.pkl')
        joblib.dump(imputer, f'models/price_prediction/{symbol}_price_imputer.pkl')
        print(f"Model saved to models/price_prediction/{symbol}_price_model.pkl")
    
    return model, scaler, imputer, mse, mae, r2, y_test, y_pred

def train_direction_model(df, symbol, save=True):
    """
    Train model for next day price direction prediction
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    save (bool): Whether to save the model
    
    Returns:
    tuple: model, scaler, imputer, accuracy, y_test, y_pred
    """
    print(f"Training direction prediction model for {symbol}...")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, imputer = prepare_data(
        df, DIRECTION_FEATURES, 'Next_Day_Direction', 'classification'
    )
    
    # Select the best model based on our previous analysis
    if symbol in ['JNJ', 'PFE']:
        model = XGBClassifier(n_estimators=100, random_state=42)
    elif symbol in ['MRK', 'ABT']:
        model = KNeighborsClassifier(n_neighbors=5)
    else:  # UNH
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model evaluation - Accuracy: {accuracy:.4f}")
    
    if save:
        # Save the model and preprocessing objects
        os.makedirs('models/direction_prediction', exist_ok=True)
        joblib.dump(model, f'models/direction_prediction/{symbol}_direction_model.pkl')
        joblib.dump(scaler, f'models/direction_prediction/{symbol}_direction_scaler.pkl')
        joblib.dump(imputer, f'models/direction_prediction/{symbol}_direction_imputer.pkl')
        print(f"Model saved to models/direction_prediction/{symbol}_direction_model.pkl")
    
    return model, scaler, imputer, accuracy, y_test, y_pred

def train_volatility_model(df, symbol, save=True):
    """
    Train model for volatility prediction
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    save (bool): Whether to save the model
    
    Returns:
    tuple: model, scaler, imputer, mse, mae, r2, y_test, y_pred
    """
    print(f"Training volatility prediction model for {symbol}...")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, imputer = prepare_data(
        df, VOLATILITY_FEATURES, 'Next_Day_Volatility', 'regression'
    )
    
    # Select the best model based on our previous analysis
    if symbol in ['JNJ', 'UNH']:
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif symbol in ['PFE', 'ABT']:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:  # MRK
        model = LinearRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model evaluation - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    if save:
        # Save the model and preprocessing objects
        os.makedirs('models/volatility_prediction', exist_ok=True)
        joblib.dump(model, f'models/volatility_prediction/{symbol}_volatility_model.pkl')
        joblib.dump(scaler, f'models/volatility_prediction/{symbol}_volatility_scaler.pkl')
        joblib.dump(imputer, f'models/volatility_prediction/{symbol}_volatility_imputer.pkl')
        print(f"Model saved to models/volatility_prediction/{symbol}_volatility_model.pkl")
    
    return model, scaler, imputer, mse, mae, r2, y_test, y_pred

def predict_next_day(symbol, price_model, price_scaler, price_imputer, 
                    direction_model, direction_scaler, direction_imputer,
                    volatility_model, volatility_scaler, volatility_imputer):
    """
    Make predictions for the next trading day
    
    Parameters:
    symbol (str): Stock symbol
    price_model, price_scaler, price_imputer: Price prediction model and preprocessing objects
    direction_model, direction_scaler, direction_imputer: Direction prediction model and preprocessing objects
    volatility_model, volatility_scaler, volatility_imputer: Volatility prediction model and preprocessing objects
    
    Returns:
    dict: Predictions for next day
    """
    # Fetch the most recent data
    df = fetch_stock_data(symbol, period='60d', save=False)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Get the latest data point
    latest_data = df.iloc[-1:][PRICE_FEATURES]
    
    # Make predictions
    # Price prediction
    latest_data_imputed = price_imputer.transform(latest_data)
    latest_data_scaled = price_scaler.transform(latest_data_imputed)
    price_pred = price_model.predict(latest_data_scaled)[0]
    
    # Direction prediction
    latest_data_imputed = direction_imputer.transform(latest_data)
    latest_data_scaled = direction_scaler.transform(latest_data_imputed)
    direction_pred = direction_model.predict(latest_data_scaled)[0]
    direction_prob = direction_model.predict_proba(latest_data_scaled)[0]
    
    # Volatility prediction
    latest_data_imputed = volatility_imputer.transform(latest_data)
    latest_data_scaled = volatility_scaler.transform(latest_data_imputed)
    volatility_pred = volatility_model.predict(latest_data_scaled)[0]
    
    # Current price
    current_price = df['Close'].iloc[-1]
    
    # Calculate expected change
    expected_change = price_pred - current_price
    expected_change_pct = (expected_change / current_price) * 100
    
    # Prepare results
    results = {
        'symbol': symbol,
        'current_date': df.index[-1].strftime('%Y-%m-%d'),
        'current_price': current_price,
        'predicted_price': price_pred,
        'expected_change': expected_change,
        'expected_change_pct': expected_change_pct,
        'direction': 'UP' if direction_pred == 1 else 'DOWN',
        'direction_probability': direction_prob[1] if direction_pred == 1 else direction_prob[0],
        'predicted_volatility': volatility_pred
    }
    
    return results

def process_stock(symbol, train=True, predict=True, visualize=True):
    """
    Process a single stock: fetch data, train models, make predictions, and visualize results
    
    Parameters:
    symbol (str): Stock symbol
    train (bool): Whether to train new models
    predict (bool): Whether to make next-day predictions
    visualize (bool): Whether to create visualizations
    
    Returns:
    dict: Results including trained models and predictions
    """
    print(f"\n{'='*50}\nProcessing {symbol} - {STOCK_DESCRIPTIONS[symbol]}\n{'='*50}")
    
    results = {'symbol': symbol}
    
    # Fetch data
    try:
        df = pd.read_csv(f'data/{symbol}_historical.csv', index_col='Date', parse_dates=True)
        print(f"Loaded data from data/{symbol}_historical.csv")
    except FileNotFoundError:
        df = fetch_stock_data(symbol, period='5y')
    
    # Calculate technical indicators
    df_processed = calculate_technical_indicators(df)
    
    # Visualize data
    if visualize:
        visualize_stock_price_history(df_processed, symbol, save_path=f'visualizations/{symbol}/price_history.png')
        visualize_technical_indicators(df_processed, symbol, save_path=f'visualizations/{symbol}/technical_indicators.png')
    
    # Train or load models
    if train:
        # Train price prediction model
        price_model, price_scaler, price_imputer, price_mse, price_mae, price_r2, price_y_test, price_y_pred = train_price_model(df_processed, symbol)
        
        # Train direction prediction model
        direction_model, direction_scaler, direction_imputer, direction_accuracy, direction_y_test, direction_y_pred = train_direction_model(df_processed, symbol)
        
        # Train volatility prediction model
        volatility_model, volatility_scaler, volatility_imputer, volatility_mse, volatility_mae, volatility_r2, volatility_y_test, volatility_y_pred = train_volatility_model(df_processed, symbol)
        
        # Visualize predictions
        if visualize:
            visualize_price_predictions(price_y_test, price_y_pred, symbol, save_path=f'visualizations/{symbol}/price_predictions.png')
            visualize_direction_predictions(direction_y_test, direction_y_pred, symbol, save_path=f'visualizations/{symbol}/direction_predictions.png')
            visualize_volatility_predictions(volatility_y_test, volatility_y_pred, symbol, save_path=f'visualizations/{symbol}/volatility_predictions.png')
        
        # Store model evaluation metrics
        results['model_evaluation'] = {
            'price': {
                'mse': price_mse,
                'mae': price_mae,
                'r2': price_r2
            },
            'direction': {
                'accuracy': direction_accuracy
            },
            'volatility': {
                'mse': volatility_mse,
                'mae': volatility_mae,
                'r2': volatility_r2
            }
        }
    else:
        # Load existing models
        try:
            price_model = joblib.load(f'models/price_prediction/{symbol}_price_model.pkl')
            price_scaler = joblib.load(f'models/price_prediction/{symbol}_price_scaler.pkl')
            price_imputer = joblib.load(f'models/price_prediction/{symbol}_price_imputer.pkl')
            
            direction_model = joblib.load(f'models/direction_prediction/{symbol}_direction_model.pkl')
            direction_scaler = joblib.load(f'models/direction_prediction/{symbol}_direction_scaler.pkl')
            direction_imputer = joblib.load(f'models/direction_prediction/{symbol}_direction_imputer.pkl')
            
            volatility_model = joblib.load(f'models/volatility_prediction/{symbol}_volatility_model.pkl')
            volatility_scaler = joblib.load(f'models/volatility_prediction/{symbol}_volatility_scaler.pkl')
            volatility_imputer = joblib.load(f'models/volatility_prediction/{symbol}_volatility_imputer.pkl')
            
            print(f"Loaded existing models for {symbol}")
        except FileNotFoundError:
            print(f"Could not find existing models for {symbol}. Please train models first.")
            return None
    
    # Make next-day predictions
    if predict:
        predictions = predict_next_day(
            symbol,
            price_model, price_scaler, price_imputer,
            direction_model, direction_scaler, direction_imputer,
            volatility_model, volatility_scaler, volatility_imputer
        )
        results['predictions'] = predictions
        
        # Print predictions
        print("\nNext Day Predictions:")
        print(f"Symbol: {predictions['symbol']}")
        print(f"Current Date: {predictions['current_date']}")
        print(f"Current Price: ${predictions['current_price']:.2f}")
        print(f"Predicted Price: ${predictions['predicted_price']:.2f}")
        print(f"Expected Change: ${predictions['expected_change']:.2f} ({predictions['expected_change_pct']:.2f}%)")
        print(f"Direction: {predictions['direction']} (Confidence: {predictions['direction_probability']*100:.2f}%)")
        print(f"Predicted Volatility: {predictions['predicted_volatility']:.4f}")
    
    return results

def main():
    """Main function to process all stocks or a specific stock"""
    parser = argparse.ArgumentParser(description='Healthcare Stocks Prediction System')
    parser.add_argument('--symbol', type=str, help='Stock symbol to process (default: process all stocks)')
    parser.add_argument('--no-train', dest='train', action='store_false', help='Skip model training and use existing models')
    parser.add_argument('--no-predict', dest='predict', action='store_false', help='Skip next-day predictions')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false', help='Skip visualizations')
    parser.set_defaults(train=True, predict=True, visualize=True)
    
    args = parser.parse_args()
    
    # Create output directories
    create_output_dirs()
    
    # Process stocks
    all_results = {}
    
    if args.symbol:
        if args.symbol in STOCK_DESCRIPTIONS:
            results = process_stock(args.symbol, train=args.train, predict=args.predict, visualize=args.visualize)
            if results:
                all_results[args.symbol] = results
        else:
            print(f"Unknown symbol: {args.symbol}")
            print(f"Available symbols: {', '.join(STOCK_DESCRIPTIONS.keys())}")
    else:
        for symbol in STOCK_DESCRIPTIONS.keys():
            results = process_stock(symbol, train=args.train, predict=args.predict, visualize=args.visualize)
            if results:
                all_results[symbol] = results
    
    # Create summary of predictions
    if args.predict and all_results:
        predictions = {symbol: results['predictions'] for symbol, results in all_results.items() if 'predictions' in results}
        
        if predictions:
            # Create a DataFrame with the predictions
            predictions_df = pd.DataFrame(predictions).T
            
            # Reorder columns for better readability
            columns_order = ['symbol', 'current_date', 'current_price', 'predicted_price', 
                            'expected_change', 'expected_change_pct', 'direction', 
                            'direction_probability', 'predicted_volatility']
            predictions_df = predictions_df[columns_order]
            
            # Format the DataFrame
            predictions_df['current_price'] = predictions_df['current_price'].round(2)
            predictions_df['predicted_price'] = predictions_df['predicted_price'].round(2)
            predictions_df['expected_change'] = predictions_df['expected_change'].round(2)
            predictions_df['expected_change_pct'] = predictions_df['expected_change_pct'].round(2)
            predictions_df['direction_probability'] = (predictions_df['direction_probability'] * 100).round(2)
            predictions_df['predicted_volatility'] = predictions_df['predicted_volatility'].round(4)
            
            # Save predictions to CSV
            predictions_file = f"predictions_{datetime.datetime.now().strftime('%Y%m%d')}.csv"
            predictions_df.to_csv(predictions_file)
            print(f"\nPredictions saved to {predictions_file}")
            
            # Print summary
            print("\nSummary of Next Day Predictions:")
            print(predictions_df)

if __name__ == "__main__":
    main()
