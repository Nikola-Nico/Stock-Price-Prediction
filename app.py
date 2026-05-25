import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from src.preprocessing import load_data_for_company
from src.features import create_features

st.set_page_config(page_title="Паметно Предвидување Акции", layout="wide")

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
    return ["NVDA", "MSFT", "AAPL", "AMZN", "GOOG", "TSLA", "META"]

available_tickers = get_all_available_tickers()

st.sidebar.header("Опции за анализа")

ticker_input = st.sidebar.selectbox(
    "Избери компанија од базата:",
    options=available_tickers,
    index=available_tickers.index("NVDA") if "NVDA" in available_tickers else 0
)

test_days = st.sidebar.slider("Приказ на денови за тест период:", min_value=30, max_value=200, value=150)
future_days = st.sidebar.slider("Денови за предвидување во ИДНИНАТА:", min_value=5, max_value=30, value=30)
start_button = st.sidebar.button("Стартувај предвидување")

st.title("Систем за предвидување на цени на акции (S&P 500)")

if start_button:
    try:
        with st.spinner(f"Се обработуваат податоците и се тренира моделот за {ticker_input}..."):
            
            # 1. Вчитување и генерирање карактеристики
            df_raw = load_data_for_company(ticker_input)
            df_features = create_features(df_raw)
            
            # Ажурирана листа со новите карактеристики од features.py
            features_list = ['MA7', 'MA21', 'EMA12', 'EMA26', 'MA_diff', 'Volatility', 'RSI']
            X = df_features[features_list]
            y = df_features['Daily_Return']  
            
            # 2. Поделба на Train/Test (За евалуација на моделот)
            split_idx = len(df_features) - test_days
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            actual_prices = df_features['Close'].iloc[split_idx:].values
            previous_prices = df_features['Close'].iloc[split_idx-1:-1].values
            
            # 3. Тренирање на XGBoost
            model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42)
            model.fit(X_train, y_train)
            
            # 4. Тестирање на историските податоци (Backtest)
            predicted_returns = model.predict(X_test)
            predicted_prices = previous_prices * (1 + predicted_returns)
            mae_dollars = mean_absolute_error(actual_prices, predicted_prices)
            
            # 5. ИДНО ПРЕДВИДУВАЊЕ (Сега ја користиме целата база за последен контекст)
            # Правиме копија од последните потребни редови за да пресметаме идни индикатори
            future_df = df_features.copy()
            
            future_predictions = []
            
            for i in range(future_days):
                # Секогаш ги земаме само карактеристиките од последниот ред
                last_row_features = future_df[features_list].iloc[-1].values.reshape(1, -1)
                
                # Моделот го предвидува процентуалниот поврат за утрешниот ден
                pred_return = model.predict(last_row_features)[0]
                
                # Ја земаме последната позната Close цена
                last_close = future_df['Close'].iloc[-1]
                
                # Пресметуваме нова цена во долари
                next_close = last_close * (1 + pred_return)
                future_predictions.append(next_close)
                
                # Креираме нов ред кој ќе го додадеме во табелата за да се пресметаат новите Moving Averages
                new_row = {
                    'Date': future_df['Date'].iloc[-1] + pd.Timedelta(days=1),
                    'Close': next_close
                }
                
                # Привремено додавање на новиот ред
                future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)
                
                # Повторно ги пресметуваме карактеристиките за да има вредности за наредниот чекор во јамката
                future_df = create_features(future_df[['Date', 'Close']])
            
            # Креирање дати за идниот период
            last_date = df_features['Date'].iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
            
            # 6. ПРИКАЗ НА РЕЗУЛТАТИ ВО STREAMLIT
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Тестирана компанија", value=ticker_input)
            with col2:
                st.metric(label="Историска MAE грешка (Тест период)", value=f"${mae_dollars:.2f}")
                
            # График 1: Историско тестирање
            st.subheader("1. Евалуација на моделот (Историски тест период)")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(actual_prices, label="Вистинска цена (Actual)", color='blue', alpha=0.8)
            ax.plot(predicted_prices, label="Предвидена цена (Predicted)", color='orange', linestyle='--', alpha=0.9)
            ax.set_title(f"Споредба на цени во тест периодот за {ticker_input}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # График 2: Предвидување во иднината
            st.subheader(f"2. Предвидување на цената за наредните {future_days} дена")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            
            # Последните 30 дена од историјата за убав визуелен премин
            history_show_days = 30
            ax2.plot(df_features['Date'].iloc[-history_show_days:], df_features['Close'].iloc[-history_show_days:], label="Историска цена", color='blue')
            ax2.plot(future_dates, future_predictions, label="Идно предвидување (Forecast)", color='green', linestyle=':', marker='o')
            
            ax2.set_title(f"Проекција на цената за {ticker_input}")
            ax2.set_xlabel("Датум")
            ax2.set_ylabel("Цена во $")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            
            # Табела со идните цени
            st.subheader("Табеларен приказ на предвидените идни цени")
            future_table = pd.DataFrame({
                'Датум': future_dates.strftime('%Y-%m-%d'),
                'Предвидена цена ($)': np.round(future_predictions, 2)
            })
            st.dataframe(future_table, use_container_width=True)
            
            st.success(f"Успешно генерирано идно предвидување за {ticker_input}!")
            
    except Exception as e:
        st.error(f"Грешка при обработката: {str(e)}")