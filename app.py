import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from src.preprocessing import load_data_for_company
from src.features import create_features

# Подесување на страницата
st.set_page_config(page_title="Паметно Предвидување Акции", layout="wide")

# Функција за автоматско наоѓање на сите достапни компании во базата
@st.cache_data
def get_all_available_tickers():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', 'data', 'raw', 'sp500_stocks.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Symbol'])
            tickers = sorted(df['Symbol'].dropna().unique().tolist())
            return tickers
    except:
        pass
    # Алтернативна листа ако нешто заглави со патеката на CSV-то
    return ["NVDA", "MSFT", "AAPL", "AMZN", "GOOG", "TSLA", "AOS"]

# Вчитај ги сите достапни симболи за менито
available_tickers = get_all_available_tickers()

# Странично мени
st.sidebar.header("Опции за анализа")

# СМЕНЕТО: Наместо text_input, сега користиме паѓачко мени (selectbox)
ticker_input = st.sidebar.selectbox(
    "Избери компанија од базата:",
    options=available_tickers,
    index=available_tickers.index("NVDA") if "NVDA" in available_tickers else 0
)

test_days = st.sidebar.slider("Приказ на денови за тест период:", min_value=30, max_value=200, value=150)
start_button = st.sidebar.button("Стартувај предвидување")

st.title("Систем за предвидување на цени на акции (S&P 500)")

if start_button:
    try:
        with st.spinner(f"Се тренира моделот во реално време за {ticker_input}..."):
            # 1. Вчитување податоци
            df_raw = load_data_for_company(ticker_input)
            
            # 2. Креирање карактеристики
            df_features = create_features(df_raw)
            
            features_list = ['MA7', 'MA21', 'Volatility']
            X = df_features[features_list]
            y = df_features['Daily_Return']  # Предвидуваме процент на промена
            
            # Сплит на податоци
            split_idx = len(df_features) - test_days
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            actual_prices = df_features['Close'].iloc[split_idx:].values
            previous_prices = df_features['Close'].iloc[split_idx-1:-1].values
            
            # 3. Тренирање на реалниот XGBoost модел во позадина
            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # 4. Предвидување проценти
            predicted_returns = model.predict(X_test)
            
            # 5. Реконструкција на цените во долари
            predicted_prices = previous_prices * (1 + predicted_returns)
            
            # 6. Пресметка на MAE грешка во долари
            mae_dollars = mean_absolute_error(actual_prices, predicted_prices)
            
            # Приказ на метрики
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Тестирана компанија", value=ticker_input)
            with col2:
                st.metric(label="Просечна MAE грешка", value=f"${mae_dollars:.2f}")
                
            # 7. Графикон
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(actual_prices, label="Вистинска цена (Actual)", color='blue', alpha=0.8)
            ax.plot(predicted_prices, label="Предвидена цена (Predicted)", color='orange', linestyle='--', alpha=0.9)
            ax.set_title(f"Конечна паметна споредба на цени за {ticker_input}", fontsize=14)
            ax.set_xlabel("Денови (Тест период)", fontsize=11)
            ax.set_ylabel("Цена во $", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            st.pyplot(fig)
            st.success(f"Успешно извршено предвидување без мемориски 'Lag' ефект за {ticker_input}!")
            
    except Exception as e:
        st.error(f"Грешка при обработката: {str(e)}")