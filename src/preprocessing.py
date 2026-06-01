import pandas as pd
import os

def load_data_for_company(ticker):
    """
    Динамички ги вчитува податоците само за внесената компанија од главната база.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Проверуваме дали базата е во соодветната папка data/raw
    file_path = os.path.join(base_dir, '..', 'data', 'raw', 'sp500_stocks.csv')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Базата со податоци не е пронајдена на патеката: {file_path}")
        
    df = pd.read_csv(file_path)
    
    company_df = df[df['Symbol'].str.upper() == ticker.upper()].copy()
    
    if company_df.empty:
        raise ValueError(f"Нема податоци за компанијата со кратенка: {ticker}")
        
    company_df['Date'] = pd.to_datetime(company_df['Date'])
    company_df = company_df.sort_values('Date').reset_index(drop=True)
    
    return company_df



# """
# preprocessing.py
# ----------------
# Вчитување, чистење и подготовка на суровите податоци.
# """

# import os
# import pandas as pd
# import numpy as np
# from src.preprocessing import load_data

# def load_data(ticker: str = 'TSLA', base_dir: str = None) -> pd.DataFrame:
#     """
#     Вчитува CSV датотека, го филтрира по тикер и го сортира хронолошки.
#     """
#     if base_dir is None:
#         base_dir = os.getcwd()

#     file_path = os.path.join(base_dir, '..', 'data', 'raw', 'sp500_stocks.csv')

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"Базата не е пронајдена на патеката: {file_path}")

#     raw_df = pd.read_csv(file_path)

#     # Чистење на празни места (spaces) од колоните и тикерите
#     raw_df.columns = raw_df.columns.str.strip()
#     raw_df['Symbol'] = raw_df['Symbol'].astype(str).str.strip()

#     # Филтрирање и хронолошко сортирање
#     df = raw_df[raw_df['Symbol'].str.upper() == ticker.upper()].copy()
#     df['Date'] = pd.to_datetime(df['Date'])
#     df = df.sort_values('Date').reset_index(drop=True)

#     print(f" Првично филтрирани редови за {ticker}: {df.shape[0]}")
#     return df


# def clean_data(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Отстранува NaN вредности и ги обработува аутлаерите преку IQR capping.
#     """
#     # Отстранување на NaN вредности во критичната колона 'Close'
#     df = df.dropna(subset=['Close']).reset_index(drop=True)
#     print(f" Редови по елиминација на празни цени: {df.shape[0]}")

#     # Пресметуваме дневен поврат бидејќи суровата цена не е стационарна
#     df['Daily_Return_Raw'] = df['Close'].pct_change().fillna(0)

#     # IQR детекција и capping на аутлаери
#     Q1 = df['Daily_Return_Raw'].quantile(0.25)
#     Q3 = df['Daily_Return_Raw'].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     outliers = df[(df['Daily_Return_Raw'] < lower_bound) | (df['Daily_Return_Raw'] > upper_bound)]
#     print(f" IQR детекција: Пронајдени се {outliers.shape[0]} денови со екстремни пазарни шокови.")

#     df['Daily_Return_Cleaned'] = df['Daily_Return_Raw'].clip(lower=lower_bound, upper=upper_bound)
#     df.drop(columns=['Daily_Return_Raw'], inplace=True)
#     print(f" Успешно извршено мазнење на аномалиите во опсег [{lower_bound:.4f}, {upper_bound:.4f}].")

#     return df


# def run_preprocessing(ticker: str = 'TSLA', base_dir: str = None) -> pd.DataFrame:
#     """
#     Главна функција - ги повикува load_data и clean_data по ред.
#     """
#     df = load_data(ticker=ticker, base_dir=base_dir)
#     df = clean_data(df)
#     return df