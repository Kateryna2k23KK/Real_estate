# 0. Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn.decomposition import PCA

# 1. Load the data
train_data = pd.read_csv('/Users/katerynayakymenko/PycharmProjects/Real_Estate/train.csv')

# 2. Initial data analysis
print(train_data.head())  # Первые 5 строк
print(train_data.info())  # Информация о столбцах
print(train_data.describe())  # Описательная статистика для числовых данных


# 3. Handle missing values
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].median())
train_data['Alley'] = train_data['Alley'].fillna(train_data['Alley'].mode()[0])
train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])
train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])
train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])
train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna(train_data['BsmtFinType1'].mode()[0])
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])
train_data['Electrical'] = train_data['Electrical'].fillna(train_data['Electrical'].mode()[0])
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])
train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])
train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])
train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])
train_data['PoolQC'] = train_data['PoolQC'].fillna(train_data['PoolQC'].mode()[0])
train_data['Fence'] = train_data['Fence'].fillna(train_data['Fence'].mode()[0])
train_data['MiscFeature'] = train_data['MiscFeature'].fillna(train_data['MiscFeature'].mode()[0])

# Remove rows with remaining missing values
train_data_cleaned = train_data.dropna()

# 4. One-Hot Encoding of categorical variables
train_data_cleaned = pd.get_dummies(train_data_cleaned)

# 5. Split the data into features and target variable
X = train_data_cleaned.drop('SalePrice', axis=1)
y = train_data_cleaned['SalePrice']

# 6. Standardize the numerical data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Outlier analysis
# Use Z-scores to identify outliers
from scipy.stats import zscore

z_scores = np.abs(zscore(X_train))
X_train_cleaned = X_train[(z_scores < 3).all(axis=1)]
y_train_cleaned = y_train[(z_scores < 3).all(axis=1)]

# 9. Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 10. Predict using Linear Regression
y_pred_lr = lr_model.predict(X_test)

# 11. Evaluate Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = mse_lr ** 0.5
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression RMSE: {rmse_lr}")
print(f"Linear Regression R²: {r2_lr}")

# 12. Cross-validation for Linear Regression
cv_lr = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Linear Regression Cross-validation MSE: {np.mean(cv_lr)}")

# 13. Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 14. Predict using Random Forest
y_pred_rf = rf_model.predict(X_test)

# 15. Train XGBoost model
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# 16. Predict using XGBoost
y_pred_xgb = xgb_model.predict(X_test)

# 17. Evaluate Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = mse_rf ** 0.5
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest RMSE: {rmse_rf}")
print(f"Random Forest R²: {r2_rf}")

# 18. Cross-validation for Random Forest
cv_rf = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Random Forest Cross-validation MSE: {np.mean(cv_rf)}")

# 19. Evaluate XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = mse_xgb ** 0.5
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost RMSE: {rmse_xgb}")
print(f"XGBoost R²: {r2_xgb}")

# 20. Model interpretation with SHAP
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

# SHAP explainer
explainer = shap.Explainer(xgb_model, X_train_df)
shap_values = explainer(X_test_df)

# Feature importance visualization
shap.summary_plot(shap_values, X_test_df)


# 21. Additional visualizations:
# 1. Actual vs Predicted Prices plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, alpha=0.7, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Prices (XGBoost)', fontsize=14)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# 2. Residuals plot
residuals = y_test - y_pred_xgb
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=50, kde=True, color='green')
plt.title('Residuals Distribution', fontsize=14)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# 3. Additional PCA visualization for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', legend='full')
plt.title('PCA of Data', fontsize=14)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# 22. Hyperparameter tuning using GridSearchCV for RandomForest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                              param_grid=rf_param_grid,
                              cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
rf_grid_search.fit(X_train, y_train)
print(f"Best parameters for Random Forest: {rf_grid_search.best_params_}")
rf_best_model = rf_grid_search.best_estimator_

# 23. Hyperparameter tuning using GridSearchCV for XGBoost
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0]
}
xgb_grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42),
                               param_grid=xgb_param_grid,
                               cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
xgb_grid_search.fit(X_train, y_train)
print(f"Best parameters for XGBoost: {xgb_grid_search.best_params_}")
xgb_best_model = xgb_grid_search.best_estimator_


# 24. Evaluate best Random Forest model after GridSearch
y_pred_rf_best = rf_best_model.predict(X_test)
mse_rf_best = mean_squared_error(y_test, y_pred_rf_best)
rmse_rf_best = mse_rf_best ** 0.5
r2_rf_best = r2_score(y_test, y_pred_rf_best)
print(f"Best Random Forest RMSE: {rmse_rf_best}")
print(f"Best Random Forest R²: {r2_rf_best}")

# 25. Evaluate best XGBoost model after GridSearch
y_pred_xgb_best = xgb_best_model.predict(X_test)
mse_xgb_best = mean_squared_error(y_test, y_pred_xgb_best)
rmse_xgb_best = mse_xgb_best ** 0.5
r2_xgb_best = r2_score(y_test, y_pred_xgb_best)
print(f"Best XGBoost RMSE: {rmse_xgb_best}")
print(f"Best XGBoost R²: {r2_xgb_best}")
