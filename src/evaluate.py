import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, y_test):
    """Пресметува MAE за сите модели."""
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        error = mean_absolute_error(y_test, preds)
        results[name] = error
    return results

def plot_predictions(y_test, preds, target_scaler, symbol):
    """Црта график на вистински наспроти предвидени цени."""
    # Враќање на цените во оригиналниот формат ($)
    real_prices = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))
    predicted_prices = target_scaler.inverse_transform(preds.reshape(-1, 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices, label="Вистинска цена (Actual)", color='blue', alpha=0.7)
    plt.plot(predicted_prices, label="Предвидена цена (Predicted)", color='orange', linestyle='--')
    plt.title(f"Споредба на цени за {symbol} - XGBoost Model")
    plt.xlabel("Денови (Тест период)")
    plt.ylabel("Цена во $")
    plt.legend()
    plt.show()


def plot_feature_importance(model, feature_columns):
    """Црта график кој покажува кои карактеристики се најважни за моделот."""
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=feature_columns).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 5))
    feat_imp.plot(kind='barh', color='teal')
    plt.title("Важност на карактеристиките (Feature Importance) - XGBoost")
    plt.xlabel("Релативна важност")
    plt.show()