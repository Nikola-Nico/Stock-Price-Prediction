"""
models.py
---------
Тренирање, евалуација и споредба на ML модели.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# 1. Train / Test Split + Scaling
# ---------------------------------------------------------------------------

def split_and_scale(df_ml: pd.DataFrame, feature_cols: list, test_ratio: float = 0.2):
    """
    Хронолошки ги дели податоците (без shuffle) и ги скалира со StandardScaler.

    Враќа:
        X_train_scaled, X_test_scaled, y_train, y_test, split_idx, scaler
    """
    X = df_ml[feature_cols]
    y = df_ml['Target_Price']

    split_idx = int(len(df_ml) * (1 - test_ratio))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f" Тренинг фаза: {X_train.shape[0]} денови  |  Тест фаза: {X_test.shape[0]} денови")
    print(" Карактеристиките се успешно скалирани.")

    return X_train_scaled, X_test_scaled, y_train, y_test, split_idx, scaler


# ---------------------------------------------------------------------------
# 2. Дефиниција на модели
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Враќа речник со сите ML модели кои ќе се тренираат.
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=7),
        "SVM": SVR(kernel='rbf', C=10.0, epsilon=0.05),
        "XGBoost": XGBRegressor(
            n_estimators=300, learning_rate=0.02, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
        ),
    }


# ---------------------------------------------------------------------------
# 3. Тренирање
# ---------------------------------------------------------------------------

def train_all_models(models: dict, X_train_scaled, y_train) -> dict:
    """
    Ги тренира сите модели и ги враќа обучените инстанци.
    """
    print("Иницијализација и тренирање на алгоритмите...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        print(f" [УСПЕШНО] Моделот '{name}' заврши со тренирање.")
    return models


# ---------------------------------------------------------------------------
# 4. Евалуација
# ---------------------------------------------------------------------------

def evaluate_models(trained_models: dict, X_test_scaled, y_test) -> pd.DataFrame:
    """
    Ги евалуира сите модели и враќа DataFrame со метриките.
    Вклучува и Naïve Baseline (yesterday = tomorrow).
    """
    results = {}

    # Naïve Baseline: предвидуваме дека утрешната цена == денешна
    naive_preds = y_test.shift(1).fillna(method='bfill')
    naive_mae   = mean_absolute_error(y_test, naive_preds)
    naive_rmse  = np.sqrt(mean_squared_error(y_test, naive_preds))
    naive_r2    = r2_score(y_test, naive_preds)
    results["Naïve Baseline (Референца)"] = {
        "MAE ($)": round(naive_mae, 2),
        "RMSE ($)": round(naive_rmse, 2),
        "R² Score (%)": round(naive_r2 * 100, 2)
    }

    for name, model in trained_models.items():
        preds = model.predict(X_test_scaled)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        results[name] = {
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "R² Score (%)": round(r2 * 100, 2)
        }

    df_comparison = pd.DataFrame(results).T
    print("\n=======================================================")
    print("     ФИНАЛЕН ИЗВЕШТАЈ ОД ЕВАЛУАЦИЈА НА СИТЕ МОДЕЛИ    ")
    print("=======================================================")
    try:
        from IPython.display import display
        display(df_comparison.sort_values(by="MAE ($)", ascending=True))
    except Exception:
        print(df_comparison.sort_values(by="MAE ($)", ascending=True))
    print("=======================================================")

    return df_comparison


# ---------------------------------------------------------------------------
# 5. Визуализации на резултати
# ---------------------------------------------------------------------------

def plot_best_model_predictions(df_ml: pd.DataFrame, trained_models: dict,
                                 df_comparison: pd.DataFrame,
                                 X_test_scaled, y_test, split_idx: int,
                                 ticker: str = 'TSLA') -> None:
    """
    Го исцртува победничкиот модел - Вистински наспроти Предвидени цени.
    """
    best_name  = df_comparison.drop("Naïve Baseline (Референца)", errors='ignore')["MAE ($)"].idxmin()
    best_model = trained_models[best_name]
    best_preds = best_model.predict(X_test_scaled)
    test_dates = df_ml['Date'].iloc[split_idx:]

    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, y_test.values,
             label="Вистински пазарни цени (Actual)", color='black', linewidth=1.5, alpha=0.7)
    plt.plot(test_dates, best_preds,
             label=f"Предвидени цени ({best_name})", color='crimson', linestyle='--', linewidth=2)
    plt.title(f"Финален Евалуационен Прозорец: Вистински vs Предвидени за {ticker}", fontsize=13)
    plt.xlabel("Датум")
    plt.ylabel("Цена во $")
    plt.legend(fontsize=11)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.show()


def plot_xgboost_feature_importance(trained_models: dict, feature_cols: list) -> None:
    """
    Прикажува Feature Importance на XGBoost моделот.
    """
    xgb = trained_models.get("XGBoost")
    if xgb is None:
        print("XGBoost модел не е пронајден.")
        return

    importance      = xgb.feature_importances_
    feat_imp_series = pd.Series(importance, index=feature_cols).sort_values(ascending=True)

    plt.figure(figsize=(10, 6))
    feat_imp_series.plot(kind='barh', color='teal', edgecolor='black', alpha=0.85)
    plt.title("Кои карактеристики имаат најголемо влијание? (XGBoost)")
    plt.xlabel("Релативно значење (Feature Importance)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6. Анализа за конкретен месец
# ---------------------------------------------------------------------------

def analyze_month(df_ml: pd.DataFrame, trained_models: dict, df_comparison: pd.DataFrame,
                  X_test_scaled, split_idx: int,
                  year: int = 2024, month: int = 1, ticker: str = 'TSLA') -> None:
    """
    Ден-по-ден анализа на предвидувањата за избран месец.
    """
    test_df = df_ml.iloc[split_idx:].copy()
    mask    = (test_df['Date'].dt.year == year) & (test_df['Date'].dt.month == month)
    month_df = test_df[mask]

    if month_df.empty:
        print(f"❌ Нема податоци за {year}-{month:02d} во тест сетот.")
        return

    month_indices = month_df.index - test_df.index[0]
    X_month_scaled = X_test_scaled[month_indices]

    winner_name  = df_comparison.drop("Naïve Baseline (Референца)", errors='ignore')["MAE ($)"].idxmin()
    winner_model = trained_models[winner_name]
    predictions  = winner_model.predict(X_month_scaled)

    report = pd.DataFrame({
        'Датум': month_df['Date'].dt.strftime('%Y-%m-%d'),
        'Вистинска Цена ($)': month_df['Target_Price'].round(2),
        'Предвидена Цена ($)': np.round(predictions, 2)
    })
    report['Разлика/Грешка ($)']  = (report['Предвидена Цена ($)'] - report['Вистинска Цена ($)']).round(2)
    report['Апсолутна Грешка ($)'] = report['Разлика/Грешка ($)'].abs()

    print(f"===========================================================")
    print(f"  РЕЗУЛТАТИ ДЕН-ПО-ДЕН ЗА {year}-{month:02d} СО {winner_name.upper()}")
    print(f"===========================================================")
    try:
        from IPython.display import display
        display(report.reset_index(drop=True))
    except Exception:
        print(report.reset_index(drop=True).to_string())

    month_mae = report['Апсолутна Грешка ($)'].mean()
    print(f"\n💡 Просечна грешка за {year}-{month:02d}: ${month_mae:.2f}")

    plt.figure(figsize=(12, 5))
    plt.plot(report['Датум'], report['Вистинска Цена ($)'],
             marker='o', label='Вистинска Цена (Actual)', color='black')
    plt.plot(report['Датум'], report['Предвидена Цена ($)'],
             marker='s', label='Предвидена Цена (Predicted)', color='crimson', linestyle='--')
    plt.title(f"Микро Анализа: {year}-{month:02d} за {ticker}")
    plt.xlabel("Датум")
    plt.ylabel("Цена во $")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()