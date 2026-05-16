import pandas as pd

def create_features(df):
    """Додава технички индикатори врз основа на цената."""
    df = df.copy()
    
    # Moving Averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    
    # Returns & Volatility (Ова ја прави цената паметна без лаг ефект!)
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=7).std()
    
    df.dropna(inplace=True)
    return df