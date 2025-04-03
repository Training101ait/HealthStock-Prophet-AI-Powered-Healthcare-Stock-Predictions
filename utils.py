import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import warnings
warnings.filterwarnings('ignore')

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

# Stock descriptions
STOCK_DESCRIPTIONS = {
    'JNJ': 'Johnson & Johnson - A leading multinational corporation specializing in pharmaceuticals, medical devices, and consumer health products.',
    'PFE': 'Pfizer Inc. - A global pharmaceutical company known for developing and manufacturing healthcare products and vaccines.',
    'MRK': 'Merck & Co. Inc. - A multinational pharmaceutical company that offers prescription medicines, vaccines, biologic therapies, and animal health products.',
    'ABT': 'Abbott Laboratories - A healthcare company providing diagnostics, medical devices, branded generic medicines, and nutritional products.',
    'UNH': 'UnitedHealth Group Incorporated - A diversified healthcare company offering health insurance and healthcare services.'
}

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

def calculate_technical_indicators(df):
    """
    Calculate various technical indicators for stock data
    
    Parameters:
    df (pandas.DataFrame): Historical stock data
    
    Returns:
    pandas.DataFrame: Stock data with technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure all required columns exist and contain numeric data
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values that might have been introduced
    df = df.ffill()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Price Rate of Change (ROC)
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Williams %R
    df['Williams_%R'] = -100 * ((high_max - df['Close']) / (high_max - low_min))
    
    # Commodity Channel Index (CCI)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    md_tp = tp.rolling(window=20).apply(lambda x: np.fabs(x - x.mean()).mean())
    df['CCI'] = (tp - ma_tp) / (0.015 * md_tp)
    
    # Momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    
    # Volatility (using standard deviation of returns)
    df['Volatility'] = df['Close'].pct_change().rolling(window=21).std() * np.sqrt(252)
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Target variable: Next day's closing price
    df['Next_Day_Close'] = df['Close'].shift(-1)
    
    # Target variable: Next day's price direction (1 if price goes up, 0 if it goes down)
    df['Next_Day_Direction'] = (df['Next_Day_Close'] > df['Close']).astype(int)
    
    # Target variable: Next day's return
    df['Next_Day_Return'] = df['Next_Day_Close'] / df['Close'] - 1
    
    # Target variable: Next day's volatility
    df['Next_Day_Volatility'] = df['Volatility'].shift(-1)
    
    # Fill NaN values with forward fill for indicators that require previous data
    df = df.fillna(method='ffill')
    
    # Only drop rows with NaN in essential columns
    essential_columns = ['Close', 'Next_Day_Close', 'Next_Day_Direction', 'Next_Day_Volatility']
    df = df.dropna(subset=essential_columns)
    
    return df

def prepare_data(df, features, target, prediction_type='regression', test_size=0.2):
    """
    Prepare data for machine learning models
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    features (list): List of feature column names
    target (str): Target column name
    prediction_type (str): Type of prediction ('regression' or 'classification')
    test_size (float): Proportion of data to use for testing
    
    Returns:
    tuple: X_train, X_test, y_train, y_test, scaler, imputer
    """
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

def visualize_stock_price_history(df, symbol, save_path=None):
    """
    Plot stock price history with moving averages
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(14, 10))
    
    # Create two subplots
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    
    # Plot price and moving averages
    ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax1.plot(df.index, df['MA50'], label='50-day MA', color='orange', alpha=0.7)
    ax1.plot(df.index, df['MA200'], label='200-day MA', color='red', alpha=0.7)
    ax1.set_title(f'{symbol} Stock Price with Moving Averages')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot volume
    ax2.bar(df.index, df['Volume'], color='green', alpha=0.5)
    ax2.set_title('Volume')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_technical_indicators(df, symbol, save_path=None):
    """
    Plot key technical indicators
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(16, 12))
    
    # Create four subplots
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # RSI plot
    ax1.plot(df.index, df['RSI'], color='purple')
    ax1.axhline(y=70, color='r', linestyle='--')
    ax1.axhline(y=30, color='g', linestyle='--')
    ax1.set_title('RSI')
    ax1.set_ylabel('RSI')
    ax1.grid(True)
    
    # MACD plot
    ax2.plot(df.index, df['MACD'], label='MACD', color='blue')
    ax2.plot(df.index, df['MACD_signal'], label='Signal Line', color='red')
    ax2.bar(df.index, df['MACD_hist'], label='Histogram', color='green', alpha=0.5)
    ax2.set_title('MACD')
    ax2.set_ylabel('MACD')
    ax2.legend()
    ax2.grid(True)
    
    # Bollinger Bands plot
    ax3.plot(df.index, df['Close'], label='Close Price', color='blue')
    ax3.plot(df.index, df['BB_upper'], label='Upper Band', color='red', alpha=0.7)
    ax3.plot(df.index, df['BB_middle'], label='Middle Band', color='orange', alpha=0.7)
    ax3.plot(df.index, df['BB_lower'], label='Lower Band', color='green', alpha=0.7)
    ax3.set_title('Bollinger Bands')
    ax3.set_ylabel('Price ($)')
    ax3.legend()
    ax3.grid(True)
    
    # Stochastic Oscillator plot
    ax4.plot(df.index, df['%K'], label='%K', color='blue')
    ax4.plot(df.index, df['%D'], label='%D', color='red')
    ax4.axhline(y=80, color='r', linestyle='--')
    ax4.axhline(y=20, color='g', linestyle='--')
    ax4.set_title('Stochastic Oscillator')
    ax4.set_ylabel('Value')
    ax4.legend()
    ax4.grid(True)
    
    plt.suptitle(f"{symbol} Technical Indicators", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_price_predictions(y_test, y_pred, symbol, save_path=None):
    """
    Plot actual vs predicted prices
    
    Parameters:
    y_test (pandas.Series): Actual prices
    y_pred (numpy.ndarray): Predicted prices
    symbol (str): Stock symbol
    save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red')
    plt.title(f'{symbol} - Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_direction_predictions(y_test, y_pred, symbol, save_path=None):
    """
    Plot confusion matrix for direction predictions
    
    Parameters:
    y_test (pandas.Series): Actual directions
    y_pred (numpy.ndarray): Predicted directions
    symbol (str): Stock symbol
    save_path (str, optional): Path to save the figure
    """
    # Create confusion matrix
    cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{symbol} - Direction Prediction Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_volatility_predictions(y_test, y_pred, symbol, save_path=None):
    """
    Plot actual vs predicted volatility
    
    Parameters:
    y_test (pandas.Series): Actual volatility
    y_pred (numpy.ndarray): Predicted volatility
    symbol (str): Stock symbol
    save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red')
    plt.title(f'{symbol} - Actual vs Predicted Volatility')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
    
    plt.show()

def create_output_dirs():
    """Create output directories for data, models, and visualizations"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/price_prediction', exist_ok=True)
    os.makedirs('models/direction_prediction', exist_ok=True)
    os.makedirs('models/volatility_prediction', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    for symbol in STOCK_DESCRIPTIONS.keys():
        os.makedirs(f'visualizations/{symbol}', exist_ok=True)
