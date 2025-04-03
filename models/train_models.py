import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create directories for models
os.makedirs('../models/price_prediction', exist_ok=True)
os.makedirs('../models/direction_prediction', exist_ok=True)
os.makedirs('../models/volatility_prediction', exist_ok=True)
os.makedirs('../models/evaluation', exist_ok=True)

# List of stocks
stocks = ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH']

# Features to use for prediction
PRICE_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 
    'MA5', 'MA10', 'MA20', 'MA50', 'MA200',
    'EMA12', 'EMA26', 'MACD', 'MACD_signal', 'MACD_hist',
    'RSI', 'BB_upper', 'BB_middle', 'BB_lower', '%K', '%D',
    'ATR', 'OBV', 'ROC', 'Williams_%R', 'CCI', 'Momentum', 'Volatility'
]

DIRECTION_FEATURES = PRICE_FEATURES.copy()
VOLATILITY_FEATURES = PRICE_FEATURES.copy()

# Function to prepare data for machine learning
def prepare_data(df, features, target, prediction_type='regression', test_size=0.2):
    """Prepare data for machine learning models"""
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Select features and target
    X = df[features]
    
    if prediction_type == 'regression':
        y = df[target]
    else:  # classification
        y = df[target].astype(int)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Split data into training and testing sets
    # For time series data, we use the last test_size portion for testing
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X_imputed[:split_idx], X_imputed[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, imputer

# Function to train and evaluate regression models for price prediction
def train_price_prediction_models(symbol):
    """Train and evaluate regression models for next day closing price prediction"""
    print(f"\nTraining price prediction models for {symbol}...")
    
    # Load processed data
    df = pd.read_csv(f'../analysis/processed_data/{symbol}_processed.csv', index_col='Date', parse_dates=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, imputer = prepare_data(
        df, PRICE_FEATURES, 'Next_Day_Close', 'regression'
    )
    
    # Define regression models to try
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = float('inf')  # Lower is better for MSE
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Check if this is the best model so far
        if mse < best_score:
            best_score = mse
            best_model = model
    
    # Save the best model
    if best_model is not None:
        model_filename = f'../models/price_prediction/{symbol}_price_model.pkl'
        scaler_filename = f'../models/price_prediction/{symbol}_price_scaler.pkl'
        imputer_filename = f'../models/price_prediction/{symbol}_price_imputer.pkl'
        
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(imputer, imputer_filename)
        
        # Find the name of the best model
        best_model_name = next(name for name, model in models.items() if model == best_model)
        print(f"Best model for {symbol} price prediction: {best_model_name} (MSE: {best_score:.4f})")
        
        # Save feature names for later use
        with open(f'../models/price_prediction/{symbol}_price_features.txt', 'w') as f:
            f.write('\n'.join(PRICE_FEATURES))
    
    # Save evaluation results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'../models/evaluation/{symbol}_price_prediction_results.csv')
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    results_df['MSE'].sort_values().plot(kind='bar')
    plt.title(f'{symbol} - MSE by Model')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig(f'../models/evaluation/{symbol}_price_prediction_mse.png')
    plt.close()
    
    return best_model, scaler, results_df

# Function to train and evaluate classification models for direction prediction
def train_direction_prediction_models(symbol):
    """Train and evaluate classification models for next day price direction prediction"""
    print(f"\nTraining direction prediction models for {symbol}...")
    
    # Load processed data
    df = pd.read_csv(f'../analysis/processed_data/{symbol}_processed.csv', index_col='Date', parse_dates=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, imputer = prepare_data(
        df, DIRECTION_FEATURES, 'Next_Day_Direction', 'classification'
    )
    
    # Define classification models to try
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = 0  # Higher is better for accuracy
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}")
        
        # Check if this is the best model so far
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
    
    # Save the best model
    if best_model is not None:
        model_filename = f'../models/direction_prediction/{symbol}_direction_model.pkl'
        scaler_filename = f'../models/direction_prediction/{symbol}_direction_scaler.pkl'
        imputer_filename = f'../models/direction_prediction/{symbol}_direction_imputer.pkl'
        
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(imputer, imputer_filename)
        
        # Find the name of the best model
        best_model_name = next(name for name, model in models.items() if model == best_model)
        print(f"Best model for {symbol} direction prediction: {best_model_name} (Accuracy: {best_score:.4f})")
        
        # Save feature names for later use
        with open(f'../models/direction_prediction/{symbol}_direction_features.txt', 'w') as f:
            f.write('\n'.join(DIRECTION_FEATURES))
        
        # Generate and save confusion matrix for the best model
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{symbol} - Confusion Matrix ({best_model_name})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'../models/evaluation/{symbol}_direction_confusion_matrix.png')
        plt.close()
    
    # Save evaluation results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'../models/evaluation/{symbol}_direction_prediction_results.csv')
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    results_df['Accuracy'].sort_values(ascending=False).plot(kind='bar')
    plt.title(f'{symbol} - Accuracy by Model')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(f'../models/evaluation/{symbol}_direction_prediction_accuracy.png')
    plt.close()
    
    return best_model, scaler, results_df

# Function to train and evaluate regression models for volatility prediction
def train_volatility_prediction_models(symbol):
    """Train and evaluate regression models for volatility prediction"""
    print(f"\nTraining volatility prediction models for {symbol}...")
    
    # Load processed data
    df = pd.read_csv(f'../analysis/processed_data/{symbol}_processed.csv', index_col='Date', parse_dates=True)
    
    # Create target: next day's volatility
    df['Next_Day_Volatility'] = df['Volatility'].shift(-1)
    df.dropna(subset=['Next_Day_Volatility'], inplace=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, imputer = prepare_data(
        df, VOLATILITY_FEATURES, 'Next_Day_Volatility', 'regression'
    )
    
    # Define regression models to try
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet': ElasticNet(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate each model
    results = {}
    best_model = None
    best_score = float('inf')  # Lower is better for MSE
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'MSE': mse,
            'MAE': mae,
            'R2': r2
        }
        
        print(f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Check if this is the best model so far
        if mse < best_score:
            best_score = mse
            best_model = model
    
    # Save the best model
    if best_model is not None:
        model_filename = f'../models/volatility_prediction/{symbol}_volatility_model.pkl'
        scaler_filename = f'../models/volatility_prediction/{symbol}_volatility_scaler.pkl'
        imputer_filename = f'../models/volatility_prediction/{symbol}_volatility_imputer.pkl'
        
        joblib.dump(best_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        joblib.dump(imputer, imputer_filename)
        
        # Find the name of the best model
        best_model_name = next(name for name, model in models.items() if model == best_model)
        print(f"Best model for {symbol} volatility prediction: {best_model_name} (MSE: {best_score:.4f})")
        
        # Save feature names for later use
        with open(f'../models/volatility_prediction/{symbol}_volatility_features.txt', 'w') as f:
            f.write('\n'.join(VOLATILITY_FEATURES))
    
    # Save evaluation results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f'../models/evaluation/{symbol}_volatility_prediction_results.csv')
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    results_df['MSE'].sort_values().plot(kind='bar')
    plt.title(f'{symbol} - MSE by Model')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.savefig(f'../models/evaluation/{symbol}_volatility_prediction_mse.png')
    plt.close()
    
    return best_model, scaler, results_df

# Function to create a summary of the best models
def create_model_summary(stocks):
    """Create a summary of the best models for each stock and prediction type"""
    summary = {
        'Price Prediction': {},
        'Direction Prediction': {},
        'Volatility Prediction': {}
    }
    
    for symbol in stocks:
        # Price prediction
        price_results = pd.read_csv(f'../models/evaluation/{symbol}_price_prediction_results.csv', index_col=0)
        best_price_model = price_results['MSE'].idxmin()
        best_price_mse = price_results.loc[best_price_model, 'MSE']
        
        # Direction prediction
        direction_results = pd.read_csv(f'../models/evaluation/{symbol}_direction_prediction_results.csv', index_col=0)
        best_direction_model = direction_results['Accuracy'].idxmax()
        best_direction_accuracy = direction_results.loc[best_direction_model, 'Accuracy']
        
        # Volatility prediction
        volatility_results = pd.read_csv(f'../models/evaluation/{symbol}_volatility_prediction_results.csv', index_col=0)
        best_volatility_model = volatility_results['MSE'].idxmin()
        best_volatility_mse = volatility_results.loc[best_volatility_model, 'MSE']
        
        # Add to summary
        summary['Price Prediction'][symbol] = {
            'Best Model': best_price_model,
            'MSE': best_price_mse
        }
        
        summary['Direction Prediction'][symbol] = {
            'Best Model': best_direction_model,
            'Accuracy': best_direction_accuracy
        }
        
        summary['Volatility Prediction'][symbol] = {
            'Best Model': best_volatility_model,
            'MSE': best_volatility_mse
        }
    
    # Create summary dataframes
    price_summary = pd.DataFrame({symbol: data for symbol, data in summary['Price Prediction'].items()}).T
    direction_summary = pd.DataFrame({symbol: data for symbol, data in summary['Direction Prediction'].items()}).T
    volatility_summary = pd.DataFrame({symbol: data for symbol, data in summary['Volatility Prediction'].items()}).T
    
    # Save summaries
    price_summary.to_csv('../models/evaluation/price_prediction_summary.csv')
    direction_summary.to_csv('../models/evaluation/direction_prediction_summary.csv')
    volatility_summary.to_csv('../models/evaluation/volatility_prediction_summary.csv')
    
    return summary

# Main execution
def main():
    print("Starting machine learning model development...")
    
    # Train models for each stock
    for symbol in stocks:
        # Price prediction
        price_model, price_scaler, price_results = train_price_prediction_models(symbol)
        
        # Direction prediction
        direction_model, direction_scaler, direction_results = train_direction_prediction_models(symbol)
        
        # Volatility prediction
        volatility_model, volatility_scaler, volatility_results = train_volatility_prediction_models(symbol)
    
    # Create model summary
    summary = create_model_summary(stocks)
    
    print("\nMachine learning model development completed successfully!")
    print("\nSummary of best models:")
    
    # Print price prediction summary
    print("\nPrice Prediction:")
    price_summary = pd.read_csv('../models/evaluation/price_prediction_summary.csv', index_col=0)
    print(price_summary)
    
    # Print direction prediction summary
    print("\nDirection Prediction:")
    direction_summary = pd.read_csv('../models/evaluation/direction_prediction_summary.csv', index_col=0)
    print(direction_summary)
    
    # Print volatility prediction summary
    print("\nVolatility Prediction:")
    volatility_summary = pd.read_csv('../models/evaluation/volatility_prediction_summary.csv', index_col=0)
    print(volatility_summary)

if __name__ == "__main__":
    main()
