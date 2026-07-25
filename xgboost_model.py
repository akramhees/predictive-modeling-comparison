import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

X_train = np.load('data/X_train.npy')
X_test = np.load('data/X_test.npy')
y_train = np.load('data/y_train.npy')
y_test = np.load('data/y_test.npy')

xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

y_pred = xgb_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== XGBOOST ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}

print("\nTuning XGBoost...")
grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {-grid.best_score_:.4f}")

best_xgb = grid.best_estimator_
y_pred_best = best_xgb.predict(X_test)

rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print(f"\nTuned XGBoost RMSE: {rmse_best:.4f}")
print(f"Tuned XGBoost R2: {r2_best:.4f}")

with open('results/xgb_results.txt', 'w') as f:
    f.write(f"Default RMSE: {rmse:.4f}\n")
    f.write(f"Default MAE: {mae:.4f}\n")
    f.write(f"Default R2: {r2:.4f}\n")
    f.write(f"Best params: {grid.best_params_}\n")
    f.write(f"Tuned RMSE: {rmse_best:.4f}\n")
    f.write(f"Tuned R2: {r2_best:.4f}\n")
