import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess data
data_main = pd.read_csv('1553768847-housing.csv')
data = data_main.copy()

# Handle missing values
imputer = SimpleImputer(strategy='median')
# imputer = SimpleImputer(fill_value=0)
data['total_bedrooms'] = imputer.fit_transform(data[['total_bedrooms']])

# Encode categorical variables
label_encoder = LabelEncoder()
data['ocean_proximity'] = label_encoder.fit_transform(data['ocean_proximity'])

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income','ocean_proximity']])
scaled_features_df = pd.DataFrame(scaled_features, columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income','ocean_proximity'])

# Split data
X = scaled_features_df
y = data['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define models
models = {
    'Linear Regression':LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

def Scores():
    model_scores = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        model_scores.append({name:{'rmse': rmse, 'mae': mae, 'r2': r2}})
    return model_scores