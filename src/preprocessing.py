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