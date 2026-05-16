from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
import joblib

def get_trained_models(X_train, y_train):
    """Тренира повеќе модели и ги враќа во речник."""
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVM": SVR(kernel='rbf', C=1.0, epsilon=0.1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models

    

def save_model(model, scaler, target_scaler, filepath="best_model.pkl"):
    """Го зачувува моделот и неговите скалери во еден фајл."""
    data_to_save = {
        "model": model,
        "scaler": scaler,
        "target_scaler": target_scaler
    }
    joblib.dump(data_to_save, filepath)
    print(f"Моделот е успешно зачуван во {filepath}")