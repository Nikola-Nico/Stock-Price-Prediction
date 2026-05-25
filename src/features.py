import pandas as pd
import numpy as np

def create_features(df):
    """Додава повеќе технички индикатори за подобар контекст на моделот."""
    df = df.copy()
    
    # Сите карактеристики се прават ВРЗ 'Close' цената
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Разлика меѓу брз и бавен просек (многу моќен сигнал)
    df['MA_diff'] = df['MA7'] - df['MA21']
    
    # Returns & Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=7).std()
    
    # RSI (Relative Strength Index) - Рачно пресметан без надворешни библиотеки
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9) # Спречуваме делење со нула
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Чистење на NaN вредностите кои се појавуваат заради прозорците (rolling)
    df.dropna(inplace=True)
    return df