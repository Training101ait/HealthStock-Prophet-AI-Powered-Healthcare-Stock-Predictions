import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import json
import os
from datetime import datetime

# Initialize API client
client = ApiClient()

# List of healthcare stocks to analyze
stocks = ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH']

# Create directory for raw data
os.makedirs('raw_data', exist_ok=True)

# Function to fetch and save stock data
def fetch_stock_data(symbol):
    print(f"Fetching data for {symbol}...")
    
    # Fetch maximum historical data
    stock_data = client.call_api('YahooFinance/get_stock_chart', query={
        'symbol': symbol,
        'interval': '1d',  # Daily data
        'range': 'max',    # Maximum available history
        'includeAdjustedClose': True
    })
    
    # Save raw data
    with open(f'raw_data/{symbol}_raw_data.json', 'w') as f:
        json.dump(stock_data, f)
    
    # Extract time series data
    result = stock_data['chart']['result'][0]
    timestamps = result['timestamp']
    dates = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps]
    
    # Extract price and volume data
    quote = result['indicators']['quote'][0]
    adjclose = result['indicators']['adjclose'][0]['adjclose'] if 'adjclose' in result['indicators'] else None
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': quote['open'],
        'High': quote['high'],
        'Low': quote['low'],
        'Close': quote['close'],
        'Volume': quote['volume']
    })
    
    # Add adjusted close if available
    if adjclose:
        df['Adj Close'] = adjclose
    
    # Set Date as index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Save to CSV
    df.to_csv(f'{symbol}_historical.csv')
    
    # Fetch company insights
    insights = client.call_api('YahooFinance/get_stock_insights', query={
        'symbol': symbol
    })
    
    # Save insights data
    with open(f'raw_data/{symbol}_insights.json', 'w') as f:
        json.dump(insights, f)
    
    return df

# Fetch data for all stocks
all_data = {}
for symbol in stocks:
    all_data[symbol] = fetch_stock_data(symbol)
    print(f"Saved data for {symbol}")

print("All stock data has been fetched and saved successfully!")
