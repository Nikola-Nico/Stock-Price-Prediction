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

# """
# feature_engineering.py
# -----------------------
# Изградба на технички индикатори и ML карактеристики.
# """

# import pandas as pd
# import numpy as np


# def build_optimized_features(data: pd.DataFrame, window_short: int = 7, window_long: int = 21) -> pd.DataFrame:
#     """
#     Генерира технички индикатори и го дефинира таргетот (следен ден цена).

#     Карактеристики:
#         - MA_Trend_Diff      : Разлика помеѓу кратка и долга движечка средина
#         - EMA_Spread         : Разлика на EMA 12 и EMA 26 (основа на MACD)
#         - Price_to_MA7_Ratio : Релативна позиција на цената наспроти MA_7
#         - Volatility         : Стандардна девијација на дневниот поврат
#         - RSI                : Relative Strength Index (14 дена)
#         - Target_Price       : Цената во следниот ден (y)
#     """
#     data = data.copy()

#     # Генерирање базни прозорци
#     data['MA_7']   = data['Close'].rolling(window=window_short).mean()
#     data['MA_21']  = data['Close'].rolling(window=window_long).mean()
#     data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
#     data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

#     # НЕКОРЕЛИРАНИ КАРАКТЕРИСТИКИ (намалување на мултиколинеарност)
#     data['MA_Trend_Diff']      = data['MA_7'] - data['MA_21']
#     data['EMA_Spread']         = data['EMA_12'] - data['EMA_26']
#     data['Price_to_MA7_Ratio'] = data['Close'] / (data['MA_7'] + 1e-9)

#     # Пазарна динамика: Волатилност
#     data['Volatility'] = data['Daily_Return_Cleaned'].rolling(window=window_short).std()

#     # RSI Индикатор (14 дена)
#     delta  = data['Close'].diff()
#     gain   = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss   = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs     = gain / (loss + 1e-9)
#     data['RSI'] = 100 - (100 / (1 + rs))

#     # ---- Напредни карактеристики (подобрување) ----

#     # MACD Signal Line
#     data['MACD']        = data['EMA_12'] - data['EMA_26']
#     data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
#     data['MACD_Hist']   = data['MACD'] - data['MACD_Signal']

#     # Bollinger Bands ширина
#     rolling_std          = data['Close'].rolling(window=20).std()
#     rolling_mean         = data['Close'].rolling(window=20).mean()
#     data['BB_Upper']     = rolling_mean + 2 * rolling_std
#     data['BB_Lower']     = rolling_mean - 2 * rolling_std
#     data['BB_Width']     = (data['BB_Upper'] - data['BB_Lower']) / (rolling_mean + 1e-9)
#     data['BB_Position']  = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'] + 1e-9)

#     # Момент на цена (Rate of Change)
#     data['ROC_5']  = data['Close'].pct_change(5)
#     data['ROC_10'] = data['Close'].pct_change(10)

#     # Volume индикатори
#     data['Volume_MA_7']    = data['Volume'].rolling(window=7).mean()
#     data['Volume_Ratio']   = data['Volume'] / (data['Volume_MA_7'] + 1e-9)

#     # ДЕФИНИРАЊЕ НА ТАРГЕТ (y_t = Цена во денот t+1)
#     data['Target_Price'] = data['Close'].shift(-1)

#     data.dropna(inplace=True)
#     return data


# def get_feature_columns() -> list:
#     """
#     Ги враќа имињата на карактеристиките (X колони) без таргет и мета-колони.
#     """
#     return [
#         'MA_Trend_Diff', 'EMA_Spread', 'Price_to_MA7_Ratio',
#         'Volatility', 'RSI',
#         'MACD', 'MACD_Signal', 'MACD_Hist',
#         'BB_Width', 'BB_Position',
#         'ROC_5', 'ROC_10',
#         'Volume_Ratio'
#     ]