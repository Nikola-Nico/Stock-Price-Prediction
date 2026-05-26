import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from src.preprocessing import load_data_for_company
from src.features import create_features

# КЛУЧЕН УВОЗ: Го повикуваме агентот директно од патеката src.agent
from src.agent import ask_agent

st.set_page_config(page_title="Паметно Предвидување Акции & AI Агент", layout="wide")

# Вчитување на тикери за селекторот во страничното мени
@st.cache_data
def get_ui_tickers():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'data', 'raw', 'sp500_stocks.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, usecols=['Symbol'])
            return sorted([str(t).strip() for t in df['Symbol'].dropna().unique().tolist()])
    except:
        pass
    return ["NVDA", "MSFT", "AAPL", "AMZN", "GOOG", "TSLA", "META"]

available_tickers = get_ui_tickers()

# --- ФУНКЦИЈА ЗА XGBOOST МОДЕЛ ШТО ЌЕ ЈА ПРАТИМЕ ДО АГЕНТОТ ---
def run_xgb_prediction(ticker_symbol: str, days_to_predict: int = 30):
    df_raw = load_data_for_company(ticker_symbol)
    df_features = create_features(df_raw)
    
    features_list = ['MA7', 'MA21', 'EMA12', 'EMA26', 'MA_diff', 'Volatility', 'RSI']
    X = df_features[features_list]
    y = df_features['Daily_Return']  
    
    split_idx = len(df_features) - 150
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    xgb_model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    future_df = df_features.copy()
    future_predictions = []
    
    for i in range(days_to_predict):
        last_row_features = future_df[features_list].iloc[-1].values.reshape(1, -1)
        pred_return = xgb_model.predict(last_row_features)[0]
        last_close = future_df['Close'].iloc[-1]
        next_close = last_close * (1 + pred_return)
        future_predictions.append(next_close)
        
        new_row = {
            'Date': future_df['Date'].iloc[-1] + pd.Timedelta(days=1),
            'Close': next_close
        }
        future_df = pd.concat([future_df, pd.DataFrame([new_row])], ignore_index=True)
        future_df = create_features(future_df[['Date', 'Close']])
        
    last_row = df_features.iloc[-1]
    return {
        "current_price": round(df_features['Close'].iloc[-1], 2),
        "predicted_future_prices": [round(p, 2) for p in future_predictions],
        "predicted_end_price": round(future_predictions[-1], 2),
        "metrics": {
            "RSI": round(last_row.get('RSI', 50), 2),
            "Volatility": round(last_row.get('Volatility', 0), 4),
            "MA7": round(last_row.get('MA7', last_row['Close']), 2),
            "MA21": round(last_row.get('MA21', last_row['Close']), 2),
            "MA_diff": round(last_row.get('MA_diff', 0), 2)
        }
    }

# СТРЕАМЛИТ ИНТЕРФЕЈС
st.title("Систем за предвидување на цени на акции (S&P 500) & Напреден AI Агент")

if "xgb_results" not in st.session_state:
    st.session_state.xgb_results = None

main_col1, main_col2 = st.columns([1.2, 0.8])

# --- ЛЕВА КОЛОНА: DASHBOARD ЗА ТЕХНИЧКА АНАЛИЗА ---
with main_col1:
    st.header("📈 Техничка Анализа & Модел")
    st.sidebar.header("Опции за анализа")
    ticker_input = st.sidebar.selectbox("Избери компанија од базата:", options=available_tickers, index=available_tickers.index("NVDA") if "NVDA" in available_tickers else 0)

    test_days = st.sidebar.slider("Приказ на денови за тест период:", min_value=30, max_value=200, value=150)
    future_days = st.sidebar.slider("Денови за предвидување во ИДНИНАТА:", min_value=5, max_value=30, value=30)
    start_button = st.sidebar.button("Стартувај предвидување")

    if start_button:
        try:
            with st.spinner(f"Се тренира моделот за {ticker_input}..."):
                df_raw = load_data_for_company(ticker_input)
                df_features = create_features(df_raw)
                
                features_list = ['MA7', 'MA21', 'EMA12', 'EMA26', 'MA_diff', 'Volatility', 'RSI']
                X = df_features[features_list]
                y = df_features['Daily_Return']  
                
                split_idx = len(df_features) - test_days
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                actual_prices = df_features['Close'].iloc[split_idx:].values
                previous_prices = df_features['Close'].iloc[split_idx-1:-1].values
                
                model = xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42)
                model.fit(X_train, y_train)
                
                predicted_returns = model.predict(X_test)
                predicted_prices = previous_prices * (1 + predicted_returns)
                mae_dollars = mean_absolute_error(actual_prices, predicted_prices)
                
                res_future = run_xgb_prediction(ticker_input, future_days)
                future_predictions = res_future['predicted_future_prices']
                
                last_date = df_features['Date'].iloc[-1]
                future_dates = pd.date_range(start=pd.to_datetime(last_date) + pd.Timedelta(days=1), periods=future_days)
                
                df_table = pd.DataFrame({
                    'Датум': [d.strftime('%Y-%m-%d') for d in future_dates],
                    'Предвидена цена ($)': future_predictions
                })
                
                st.session_state.xgb_results = {
                    "ticker": ticker_input,
                    "mae": mae_dollars,
                    "actual_prices": actual_prices,
                    "predicted_prices": predicted_prices,
                    "df_features": df_features,
                    "future_dates": future_dates,
                    "future_predictions": future_predictions,
                    "future_days": future_days,
                    "df_table": df_table
                }
        except ValueError as ve:
            st.error(f"⚠️ **Нема доволно податоци:**  Ве молиме намалете го тест периодот во менито лево или изберете друга компанија.")
        except Exception as e:
            # Ова ја фаќа оригиналната Scikit-Learn грешка ако некако се протне низ првата заштита
            if "0 sample(s)" in str(e):
                st.error("⚠️ **Грешка при тренинг:** Нема доволно историски податоци во базата за да се изврши XGBoost математичката анализа за оваа компанија. Пробајте со помал тест период.")
            else:
                st.error(f"Грешка: {str(e)}")

    if st.session_state.xgb_results is not None:
        res = st.session_state.xgb_results
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Тестирана компанија", value=res["ticker"])
        with m_col2:
            st.metric(label="Историска MAE грешка", value=f"${res['mae']:.2f}")
        
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(res["actual_prices"], label="Вистинска цена", color='blue')
        ax.plot(res["predicted_prices"], label="Предвидена цена", color='orange', linestyle='--')
        ax.legend(); ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        st.subheader(f"Проекција за наредните {res['future_days']} дена")
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.plot(pd.to_datetime(res["df_features"]['Date'].tail(30)), res["df_features"]['Close'].tail(30), label="Историја")
        ax2.plot(res["future_dates"], res["future_predictions"], label="Идно предвидување", color='green', marker='o')
        ax2.legend(); ax2.grid(True, alpha=0.3); plt.xticks(rotation=45)
        st.pyplot(fig2)
        
        st.subheader("Табеларен приказ на предвидените идни цени")
        st.dataframe(res["df_table"], use_container_width=True)
        st.success(f"Успешно генерирано идно предвидување за {res['ticker']}!")

# --- ДЕСНА КОЛОНА: ЧАТ АГЕНТ ОД ПОСЕБЕН ФАЈЛ (SRC/AGENT.PY) ---
with main_col2:
    st.header("💬 AI Финансиски Асистент")
    st.write("Постави прашање. Агентот работи со агенто од `Brainster` и има пристап до глобалната берза!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_container = st.container(height=450)
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    if user_input := st.chat_input("Напиши прашање (на пр. 'Дали ќе расте NVDA?' или 'Кои се топ 3 компании?')"):
        with chat_container:
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.chat_message("assistant"):
                with st.spinner("Агентот ја повикува соодветната математичка алатка од својот фајл..."):
                    # Го повикуваме одделениот агент и му ја предаваме функцијата за XGBoost предвидување
                    agent_response = ask_agent(user_input, xgb_predictor_func=run_xgb_prediction)
                    st.write(agent_response)
            st.session_state.chat_history.append({"role": "assistant", "content": agent_response})