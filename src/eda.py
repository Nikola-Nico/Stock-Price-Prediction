"""
eda.py
------
Exploratory Data Analysis - визуализации и статистики.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_and_returns(df: pd.DataFrame, ticker: str = 'TSLA') -> None:
    """
    Линеарен график на историска цена + boxplot на дневни приноси.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].plot(df['Date'], df['Close'], color='tab:blue', linewidth=1.5, label='Close Price')
    ax[0].set_title(f"Историски развој на цената за {ticker}", fontsize=12)
    ax[0].set_ylabel("Цена во $")
    ax[0].set_xlabel("Година")

    sns.boxplot(x=df['Daily_Return_Cleaned'], ax=ax[1], color='salmon')
    ax[1].set_title("Детекција на екстремни пазарни аномалии (Daily Returns)", fontsize=12)
    ax[1].set_xlabel("Процент на дневен поврат")

    plt.tight_layout()
    plt.show()


def print_data_info(df: pd.DataFrame) -> None:
    """
    Печати основни инфо, missing values и дескриптивна статистика.
    """
    print("--- Проверка на типови на податоци ---")
    print(df.info())
    print("\n--- Проверка за преостанати Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Дескриптивна статистика (Мерки на централна тенденција) ---")
    display(df.describe())


def plot_correlation_matrix(df_ml: pd.DataFrame, ticker: str = 'TSLA') -> None:
    """
    Прикажува корелациска матрица на оптимизираните карактеристики.
    """
    optimized_features = [
        'Close', 'MA_Trend_Diff', 'EMA_Spread', 'Price_to_MA7_Ratio',
        'Volatility', 'RSI', 'Target_Price'
    ]
    # Задржуваме само колони кои постојат
    cols = [c for c in optimized_features if c in df_ml.columns]

    plt.figure(figsize=(9, 7))
    corr_matrix = df_ml[cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                vmin=-1, vmax=1, linewidths=0.5)
    plt.title(f"Оптимизирана Корелациска Матрица за {ticker}")
    plt.show()


def plot_train_test_split(df_ml: pd.DataFrame, y_train, y_test, split_idx: int) -> None:
    """
    Визуелен приказ на хронолошката поделба на тренинг/тест сет.
    """
    plt.figure(figsize=(14, 4))
    plt.plot(df_ml['Date'].iloc[:split_idx], y_train,
             label='Прозорци за Тренирање (Train Set)', color='royalblue')
    plt.plot(df_ml['Date'].iloc[split_idx:], y_test,
             label='Прозорци за Тестирање (Test Set)', color='darkorange')
    plt.title("Хронолошка распределба на податочниот сет")
    plt.ylabel("Цена ($)")
    plt.legend()
    plt.show()