import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import glob

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_style('darkgrid')

# Create directories for outputs
os.makedirs('../analysis/figures', exist_ok=True)
os.makedirs('../analysis/processed_data', exist_ok=True)

# List of stocks
stocks = ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH']

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    """Calculate various technical indicators for stock data"""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure all required columns exist and contain numeric data
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill any NaN values that might have been introduced
    df = df.ffill()  # Using ffill instead of deprecated fillna(method='ffill')
    
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
    
    # Fill NaN values with forward fill for indicators that require previous data
    df = df.fillna(method='ffill')
    
    # Only drop rows with NaN in essential columns
    # This is more selective than dropping all NaN values
    essential_columns = ['Close', 'Next_Day_Close', 'Next_Day_Direction']
    df = df.dropna(subset=essential_columns)
    
    # For monthly data, we need to keep at least some rows
    if len(df) < 10:
        print(f"Warning: Very few rows after processing. Keeping data with some NaN values.")
        df = df.copy()  # Just to ensure we're not modifying the original
    
    return df

# Function to visualize stock data
def visualize_stock_data(df, symbol):
    """Create visualizations for stock data analysis"""
    # Create a figure directory for this stock
    stock_fig_dir = f'../analysis/figures/{symbol}'
    os.makedirs(stock_fig_dir, exist_ok=True)
    
    # Plot 1: Stock Price History with Moving Averages
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['MA50'], label='50-day MA', alpha=0.7)
    plt.plot(df.index, df['MA200'], label='200-day MA', alpha=0.7)
    plt.title(f'{symbol} Stock Price History with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/price_history_ma.png')
    plt.close()
    
    # Plot 2: Daily Returns Distribution
    plt.figure(figsize=(14, 7))
    sns.histplot(df['Daily_Return'].dropna(), kde=True, bins=100)
    plt.title(f'{symbol} Daily Returns Distribution')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/returns_distribution.png')
    plt.close()
    
    # Plot 3: Volatility Over Time
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Volatility'])
    plt.title(f'{symbol} Volatility Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Annualized)')
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/volatility.png')
    plt.close()
    
    # Plot 4: RSI Indicator
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['RSI'])
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title(f'{symbol} Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/rsi.png')
    plt.close()
    
    # Plot 5: MACD
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal Line')
    plt.bar(df.index, df['MACD_hist'], label='Histogram', alpha=0.5)
    plt.title(f'{symbol} MACD Indicator')
    plt.xlabel('Date')
    plt.ylabel('MACD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/macd.png')
    plt.close()
    
    # Plot 6: Bollinger Bands
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['BB_upper'], label='Upper Band', alpha=0.7)
    plt.plot(df.index, df['BB_middle'], label='Middle Band', alpha=0.7)
    plt.plot(df.index, df['BB_lower'], label='Lower Band', alpha=0.7)
    plt.title(f'{symbol} Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/bollinger_bands.png')
    plt.close()
    
    # Plot 7: Correlation Matrix of Technical Indicators
    corr_columns = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'Volatility', 'OBV', 'ROC', 'CCI', 'Momentum']
    corr_matrix = df[corr_columns].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'{symbol} Correlation Matrix of Technical Indicators')
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/correlation_matrix.png')
    plt.close()
    
    # Plot 8: Next Day Direction Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Next_Day_Direction', data=df)
    plt.title(f'{symbol} Next Day Price Direction Distribution')
    plt.xlabel('Direction (1=Up, 0=Down)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(f'{stock_fig_dir}/next_day_direction.png')
    plt.close()
    
    print(f"Visualizations for {symbol} saved to {stock_fig_dir}")

# Function to analyze correlations between stocks
def analyze_stock_correlations(all_stocks_data):
    """Analyze correlations between different stocks"""
    # Extract closing prices for all stocks
    close_prices = pd.DataFrame()
    for symbol, data in all_stocks_data.items():
        close_prices[symbol] = data['Close']
    
    # Calculate correlation matrix
    corr_matrix = close_prices.corr()
    
    # Visualize correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Healthcare Stocks')
    plt.tight_layout()
    plt.savefig('../analysis/figures/stock_correlation_matrix.png')
    plt.close()
    
    # Calculate returns correlation
    returns = close_prices.pct_change().dropna()
    returns_corr = returns.corr()
    
    # Visualize returns correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(returns_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Healthcare Stock Returns')
    plt.tight_layout()
    plt.savefig('../analysis/figures/stock_returns_correlation_matrix.png')
    plt.close()
    
    print("Stock correlation analysis completed")
    
    return corr_matrix, returns_corr

# Main execution
def main():
    print("Starting stock data analysis...")
    
    # Load data for all stocks
    all_stocks_data = {}
    for symbol in stocks:
        try:
            file_path = f'{symbol}_historical.csv'
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            
            # Check if data is monthly or daily
            print(f"Loaded data for {symbol} with {len(df)} rows")
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            all_stocks_data[symbol] = df
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
    
    # Calculate technical indicators for each stock
    processed_data = {}
    for symbol, df in all_stocks_data.items():
        print(f"Calculating technical indicators for {symbol}...")
        processed_df = calculate_technical_indicators(df)
        
        # Check if we have data after processing
        print(f"Processed data for {symbol} has {len(processed_df)} rows")
        
        processed_data[symbol] = processed_df
        
        # Save processed data
        processed_df.to_csv(f'../analysis/processed_data/{symbol}_processed.csv')
        
        # Visualize the data
        print(f"Creating visualizations for {symbol}...")
        visualize_stock_data(processed_df, symbol)
    
    # Analyze correlations between stocks
    print("Analyzing correlations between stocks...")
    price_corr, returns_corr = analyze_stock_correlations(all_stocks_data)
    
    # Save correlation matrices
    price_corr.to_csv('../analysis/processed_data/price_correlation_matrix.csv')
    returns_corr.to_csv('../analysis/processed_data/returns_correlation_matrix.csv')
    
    print("Stock data analysis completed successfully!")

if __name__ == "__main__":
    main()
