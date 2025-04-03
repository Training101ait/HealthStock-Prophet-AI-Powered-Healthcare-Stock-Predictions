import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
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

# Set page configuration
st.set_page_config(
    page_title="Healthcare Stocks Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define constants
STOCKS = ['JNJ', 'PFE', 'MRK', 'ABT', 'UNH']

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('models/price_prediction', exist_ok=True)
os.makedirs('models/direction_prediction', exist_ok=True)
os.makedirs('models/volatility_prediction', exist_ok=True)

# Function to fetch stock data
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(symbol, period='5y'):
    """
    Fetch historical stock data using Yahoo Finance API
    
    Parameters:
    symbol (str): Stock symbol
    period (str): Time period to fetch data for (default: '5y' for 5 years)
    
    Returns:
    pandas.DataFrame: Historical stock data
    """
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    
    # Reset index to make Date a column
    df = df.reset_index()
    
    # Convert Date column to datetime if it's not already
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index again
    df = df.set_index('Date')
    
    return df

# Function to train price prediction model
def train_price_model(df, symbol):
    """
    Train model for next day closing price prediction
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    
    Returns:
    tuple: model, scaler, imputer
    """
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
    
    return model, scaler, imputer

# Function to train direction prediction model
def train_direction_model(df, symbol):
    """
    Train model for next day price direction prediction
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    
    Returns:
    tuple: model, scaler, imputer
    """
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
    
    return model, scaler, imputer

# Function to train volatility prediction model
def train_volatility_model(df, symbol):
    """
    Train model for volatility prediction
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    
    Returns:
    tuple: model, scaler, imputer
    """
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
    
    return model, scaler, imputer

# Function to make predictions for the next trading day
def predict_next_day(symbol, df_processed, price_model, price_scaler, price_imputer, 
                    direction_model, direction_scaler, direction_imputer,
                    volatility_model, volatility_scaler, volatility_imputer):
    """
    Make predictions for the next trading day
    
    Parameters:
    symbol (str): Stock symbol
    df_processed (pandas.DataFrame): Processed stock data with technical indicators
    price_model, price_scaler, price_imputer: Price prediction model and preprocessing objects
    direction_model, direction_scaler, direction_imputer: Direction prediction model and preprocessing objects
    volatility_model, volatility_scaler, volatility_imputer: Volatility prediction model and preprocessing objects
    
    Returns:
    dict: Predictions for next day
    """
    # Get the latest data point
    latest_data = df_processed.iloc[-1:][PRICE_FEATURES]
    
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
    current_price = df_processed['Close'].iloc[-1]
    
    # Calculate expected change
    expected_change = price_pred - current_price
    expected_change_pct = (expected_change / current_price) * 100
    
    # Prepare results
    results = {
        'symbol': symbol,
        'current_date': df_processed.index[-1].strftime('%Y-%m-%d'),
        'current_price': current_price,
        'predicted_price': price_pred,
        'expected_change': expected_change,
        'expected_change_pct': expected_change_pct,
        'direction': 'UP' if direction_pred == 1 else 'DOWN',
        'direction_probability': direction_prob[1] if direction_pred == 1 else direction_prob[0],
        'predicted_volatility': volatility_pred
    }
    
    return results

# Function to plot stock price history
def plot_stock_price_history(df, symbol):
    """
    Plot stock price history with moving averages using Plotly
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=(f'{symbol} Stock Price with Moving Averages', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    # Add price and moving averages
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='50-day MA', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='200-day MA', line=dict(color='red')), row=1, col=1)
    
    # Add volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker=dict(color='green', opacity=0.5)), row=2, col=1)
    
    # Update layout
    fig.update_layout(height=600, title_text=f"{symbol} - {STOCK_DESCRIPTIONS[symbol]}",
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    return fig

# Function to plot technical indicators
def plot_technical_indicators(df, symbol):
    """
    Plot key technical indicators using Plotly
    
    Parameters:
    df (pandas.DataFrame): Stock data with technical indicators
    symbol (str): Stock symbol
    
    Returns:
    plotly.graph_objects.Figure: Plotly figure
    """
    # Create subplots: 2 rows, 2 columns
    fig = make_subplots(rows=2, cols=2, subplot_titles=('RSI', 'MACD', 'Bollinger Bands', 'Stochastic Oscillator'))
    
    # RSI plot
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[70]*len(df), name='Overbought', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=[30]*len(df), name='Oversold', line=dict(color='green', dash='dash')), row=1, col=1)
    
    # MACD plot
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], name='Signal Line', line=dict(color='red')), row=1, col=2)
    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='Histogram', marker=dict(color='green')), row=1, col=2)
    
    # Bollinger Bands plot
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='Upper Band', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_middle'], name='Middle Band', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='Lower Band', line=dict(color='green')), row=2, col=1)
    
    # Stochastic Oscillator plot
    fig.add_trace(go.Scatter(x=df.index, y=df['%K'], name='%K', line=dict(color='blue')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['%D'], name='%D', line=dict(color='red')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=[80]*len(df), name='Overbought', line=dict(color='red', dash='dash')), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=[20]*len(df), name='Oversold', line=dict(color='green', dash='dash')), row=2, col=2)
    
    # Update layout
    fig.update_layout(height=800, title_text=f"{symbol} Technical Indicators",
                     showlegend=False)
    
    return fig

# Function to plot predictions summary
def plot_predictions_summary(predictions_df):
    """
    Plot summary of predictions
    
    Parameters:
    predictions_df (pandas.DataFrame): DataFrame with predictions
    
    Returns:
    tuple: Three plotly figures
    """
    # Create a bar chart for expected price changes
    fig1 = px.bar(predictions_df, x='Symbol', y='Expected Change (%)', 
                color='Direction',
                color_discrete_map={'UP': 'green', 'DOWN': 'red'},
                title='Expected Price Change (%) for Next Trading Day',
                text='Expected Change (%)')
    
    fig1.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig1.update_layout(height=500)
    
    # Create a bar chart for prediction confidence
    fig2 = px.bar(predictions_df, x='Symbol', y='Confidence (%)', 
                color='Direction',
                color_discrete_map={'UP': 'green', 'DOWN': 'red'},
                title='Prediction Confidence (%) for Next Trading Day',
                text='Confidence (%)')
    
    fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig2.update_layout(height=500)
    
    # Create a bar chart for predicted volatility
    fig3 = px.bar(predictions_df, x='Symbol', y='Predicted Volatility', 
                title='Predicted Volatility for Next Trading Day',
                text='Predicted Volatility')
    
    fig3.update_traces(texttemplate='%{text:.4f}', textposition='outside', marker_color='purple')
    fig3.update_layout(height=500)
    
    return fig1, fig2, fig3

# Main app function
def main():
    # Add a title and description
    st.title("Healthcare Stocks Prediction System")
    st.markdown("""
    This app predicts the next day's closing price, price movement direction, and volatility for five major healthcare stocks:
    - Johnson & Johnson (JNJ)
    - Pfizer Inc. (PFE)
    - Merck & Co. Inc. (MRK)
    - Abbott Laboratories (ABT)
    - UnitedHealth Group Incorporated (UNH)
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Data period selection
    data_period = st.sidebar.selectbox(
        "Select data period",
        options=["1y", "2y", "5y", "10y", "max"],
        index=2  # Default to 5y
    )
    
    # Model training option
    train_models = st.sidebar.checkbox("Train new models", value=False)
    
    # Stock selection
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to analyze",
        options=STOCKS,
        default=STOCKS
    )
    
    if not selected_stocks:
        st.warning("Please select at least one stock to analyze.")
        return
    
    # Add a button to trigger the analysis
    if st.sidebar.button("Run Analysis"):
        # Show a spinner while processing
        with st.spinner("Fetching data and making predictions..."):
            # Process each selected stock
            all_predictions = {}
            
            for symbol in selected_stocks:
                st.subheader(f"Processing {symbol} - {STOCK_DESCRIPTIONS[symbol]}")
                
                # Fetch data
                df = fetch_stock_data(symbol, period=data_period)
                
                # Calculate technical indicators
                df_processed = calculate_technical_indicators(df)
                
                # Display stock price history
                st.plotly_chart(plot_stock_price_history(df_processed, symbol), use_container_width=True)
                
                # Display technical indicators
                st.plotly_chart(plot_technical_indicators(df_processed, symbol), use_container_width=True)
                
                # Train or load models
                if train_models:
                    st.info(f"Training new models for {symbol}...")
                    
                    # Train price prediction model
                    price_model, price_scaler, price_imputer = train_price_model(df_processed, symbol)
                    
                    # Train direction prediction model
                    direction_model, direction_scaler, direction_imputer = train_direction_model(df_processed, symbol)
                    
                    # Train volatility prediction model
                    volatility_model, volatility_scaler, volatility_imputer = train_volatility_model(df_processed, symbol)
                    
                    # Save models
                    joblib.dump(price_model, f'models/price_prediction/{symbol}_price_model.pkl')
                    joblib.dump(price_scaler, f'models/price_prediction/{symbol}_price_scaler.pkl')
                    joblib.dump(price_imputer, f'models/price_prediction/{symbol}_price_imputer.pkl')
                    
                    joblib.dump(direction_model, f'models/direction_prediction/{symbol}_direction_model.pkl')
                    joblib.dump(direction_scaler, f'models/direction_prediction/{symbol}_direction_scaler.pkl')
                    joblib.dump(direction_imputer, f'models/direction_prediction/{symbol}_direction_imputer.pkl')
                    
                    joblib.dump(volatility_model, f'models/volatility_prediction/{symbol}_volatility_model.pkl')
                    joblib.dump(volatility_scaler, f'models/volatility_prediction/{symbol}_volatility_scaler.pkl')
                    joblib.dump(volatility_imputer, f'models/volatility_prediction/{symbol}_volatility_imputer.pkl')
                    
                    st.success(f"Models trained and saved for {symbol}")
                else:
                    try:
                        # Load existing models
                        price_model = joblib.load(f'models/price_prediction/{symbol}_price_model.pkl')
                        price_scaler = joblib.load(f'models/price_prediction/{symbol}_price_scaler.pkl')
                        price_imputer = joblib.load(f'models/price_prediction/{symbol}_price_imputer.pkl')
                        
                        direction_model = joblib.load(f'models/direction_prediction/{symbol}_direction_model.pkl')
                        direction_scaler = joblib.load(f'models/direction_prediction/{symbol}_direction_scaler.pkl')
                        direction_imputer = joblib.load(f'models/direction_prediction/{symbol}_direction_imputer.pkl')
                        
                        volatility_model = joblib.load(f'models/volatility_prediction/{symbol}_volatility_model.pkl')
                        volatility_scaler = joblib.load(f'models/volatility_prediction/{symbol}_volatility_scaler.pkl')
                        volatility_imputer = joblib.load(f'models/volatility_prediction/{symbol}_volatility_imputer.pkl')
                        
                        st.info(f"Loaded existing models for {symbol}")
                    except FileNotFoundError:
                        st.error(f"Could not find existing models for {symbol}. Training new models...")
                        
                        # Train price prediction model
                        price_model, price_scaler, price_imputer = train_price_model(df_processed, symbol)
                        
                        # Train direction prediction model
                        direction_model, direction_scaler, direction_imputer = train_direction_model(df_processed, symbol)
                        
                        # Train volatility prediction model
                        volatility_model, volatility_scaler, volatility_imputer = train_volatility_model(df_processed, symbol)
                        
                        # Save models
                        joblib.dump(price_model, f'models/price_prediction/{symbol}_price_model.pkl')
                        joblib.dump(price_scaler, f'models/price_prediction/{symbol}_price_scaler.pkl')
                        joblib.dump(price_imputer, f'models/price_prediction/{symbol}_price_imputer.pkl')
                        
                        joblib.dump(direction_model, f'models/direction_prediction/{symbol}_direction_model.pkl')
                        joblib.dump(direction_scaler, f'models/direction_prediction/{symbol}_direction_scaler.pkl')
                        joblib.dump(direction_imputer, f'models/direction_prediction/{symbol}_direction_imputer.pkl')
                        
                        joblib.dump(volatility_model, f'models/volatility_prediction/{symbol}_volatility_model.pkl')
                        joblib.dump(volatility_scaler, f'models/volatility_prediction/{symbol}_volatility_scaler.pkl')
                        joblib.dump(volatility_imputer, f'models/volatility_prediction/{symbol}_volatility_imputer.pkl')
                        
                        st.success(f"Models trained and saved for {symbol}")
                
                # Make predictions
                predictions = predict_next_day(
                    symbol, df_processed,
                    price_model, price_scaler, price_imputer,
                    direction_model, direction_scaler, direction_imputer,
                    volatility_model, volatility_scaler, volatility_imputer
                )
                
                all_predictions[symbol] = predictions
                
                # Display predictions
                st.subheader(f"Next Day Predictions for {symbol}")
                
                # Create three columns for the metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Price",
                        value=f"${predictions['predicted_price']:.2f}",
                        delta=f"{predictions['expected_change_pct']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        label="Direction",
                        value=predictions['direction'],
                        delta=f"Confidence: {predictions['direction_probability']*100:.2f}%"
                    )
                
                with col3:
                    st.metric(
                        label="Predicted Volatility",
                        value=f"{predictions['predicted_volatility']:.4f}"
                    )
                
                st.markdown("---")
            
            # Create a DataFrame with all predictions
            if all_predictions:
                predictions_df = pd.DataFrame(all_predictions).T
                
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
                
                # Rename columns for better readability
                predictions_df = predictions_df.rename(columns={
                    'symbol': 'Symbol',
                    'current_date': 'Current Date',
                    'current_price': 'Current Price ($)',
                    'predicted_price': 'Predicted Price ($)',
                    'expected_change': 'Expected Change ($)',
                    'expected_change_pct': 'Expected Change (%)',
                    'direction': 'Direction',
                    'direction_probability': 'Confidence (%)',
                    'predicted_volatility': 'Predicted Volatility'
                })
                
                # Display the predictions table
                st.subheader("Summary of Next Day Predictions")
                st.dataframe(predictions_df)
                
                # Plot predictions summary
                fig1, fig2, fig3 = plot_predictions_summary(predictions_df)
                
                st.plotly_chart(fig1, use_container_width=True)
                st.plotly_chart(fig2, use_container_width=True)
                st.plotly_chart(fig3, use_container_width=True)
                
                # Save predictions to CSV
                predictions_file = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
                predictions_df.to_csv(predictions_file)
                
                # Add a download button
                with open(predictions_file, 'rb') as f:
                    st.download_button(
                        label="Download Predictions CSV",
                        data=f,
                        file_name=predictions_file,
                        mime="text/csv"
                    )
    
    # Add information about the app
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info("""
    This app uses machine learning to predict stock movements for major healthcare companies.
    
    The predictions are based on historical data and technical indicators, and should be used as one of many tools for investment decisions.
    
    **Note:** Past performance is not indicative of future results. Always do your own research before making investment decisions.
    """)

if __name__ == "__main__":
    main()
