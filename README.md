# Predicting-_Furniture_Sales
This project focuses on  predicting the number of furniture items sold based on various product  attributes such as product title, original price, discounted price, and tag text.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("ecommerce_furniture_dataset_2024.csv")  # Replace with actual dataset

df.head(7)

df.isnull().sum()

# Convert price columns to numeric
df['originalPrice'] = df['originalPrice'].astype(str).str.replace(r'[$,]', '', regex=True)
df['originalPrice'] = pd.to_numeric(df['originalPrice'], errors='coerce')
df['originalPrice'].fillna(df['originalPrice'].median(), inplace=True)
df['price'] = df['price'].astype(str).str.replace(r'[$,]', '', regex=True)
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['price'].fillna(df['price'].median(), inplace=True)


plt.figure(figsize=(8, 6))
sns.heatmap(df[['originalPrice', 'price', 'sold']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Price, Original Price, and Items Sold")
plt.show()

# Feature Engineering
df['discount'] = np.where(df['originalPrice'] > 0, (df['originalPrice'] - df['price']) / df['originalPrice'] * 100, 0)
df['price_diff'] = df['originalPrice'] - df['price']
df['sold'] = df['sold'].clip(lower=0)  # Prevent negative values
df['sold'] = df['sold'].clip(lower=1)  # Ensure no zeros before log
df['log_sold'] = np.log(df['sold'])  # Apply log transformation to handle skewness

tfidf = TfidfVectorizer(max_features=100)
df['productTitle'] = df['productTitle'].fillna("").astype(str)
df['tagText'] = df['tagText'].fillna("").astype(str)
title_features = tfidf.fit_transform(df['productTitle'] + " " + df['tagText']).toarray()
title_df = pd.DataFrame(title_features, columns=[f'title_{i}' for i in range(title_features.shape[1])])

# Combine features
X = pd.concat([df[['originalPrice', 'price', 'discount', 'price_diff']], title_df], axis=1)
X.replace([np.inf, -np.inf], np.nan, inplace=True)  # Convert inf → NaN
X.fillna(0, inplace=True)  # Replace NaNs with 0
y = df['log_sold']

y.head(3)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[['originalPrice', 'price', 'discount']] = scaler.fit_transform(X_train[['originalPrice', 'price', 'discount']])
X_test[['originalPrice', 'price', 'discount']] = scaler.transform(X_test[['originalPrice', 'price', 'discount']])

# Train RandomForest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

model.score(X_test,y_test)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor

# Model with Hyperparameter Tuning
param_grid = {
    'max_iter': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

model = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=42), param_grid, cv=3, n_iter=5, n_jobs=-1)
model.fit(X_train, y_train)

model.score(X_test,y_test)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Ridge Model

from sklearn.linear_model import Ridge
import lightgbm as lgb

X.isnull().sum()

df.head(3)


# Scale numerical features
scaler = StandardScaler()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_test = scaler.transform(X_test)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train LightGBM Model
lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=7, random_state=42)
lgb_model.fit(X_train, y_train)

# Train Linear Regression as Baseline
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Predictions
y_pred_lgb = lgb_model.predict(X_test)
y_pred_ridge = ridge.predict(X_test)

# Convert back from log scale
# Replace NaN and infinite values in predictions and actual values
y_test_actual = np.expm1(y_test)  # Convert back from log scale
y_pred_lgb = np.nan_to_num(y_pred_lgb, nan=0, posinf=0, neginf=0)
y_pred_ridge = np.nan_to_num(y_pred_ridge, nan=np.median(y_pred_ridge), posinf=np.median(y_pred_ridge), neginf=0)
y_pred_lgb = np.expm1(y_pred_lgb)
y_pred_ridge = np.expm1(y_pred_ridge)
y_test_actual = np.expm1(y_test)

# Evaluate Models
mae_lgb = mean_absolute_error(y_test_actual, y_pred_lgb)
mse_lgb = mean_squared_error(y_test_actual, y_pred_lgb)
r2_lgb = r2_score(y_test_actual, y_pred_lgb)

mae_ridge = mean_absolute_error(y_test_actual, y_pred_ridge)
mse_ridge = mean_squared_error(y_test_actual, y_pred_ridge)
r2_ridge = r2_score(y_test_actual, y_pred_ridge)

print("LightGBM Performance:")
print(f"MAE: {mae_lgb}")
print(f"MSE: {mse_lgb}")
print(f"R² Score: {r2_lgb}")

print("\nRidge Regression Performance:")
print(f"MAE: {mae_ridge}")
print(f"MSE: {mse_ridge}")
print(f"R² Score: {r2_ridge}")

# LinearRegression
from sklearn.linear_model import LinearRegression

# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(n_estimators=100,random_state=42)

# Train models
lr_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predict with Linear Regression
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Predict with Random Forest
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# Print model evaluation results
print(f'Linear Regression MSE: {mse_lr}, R2: {r2_lr}')
print(f'Random Forest MSE: {mse_rf}, R2: {r2_rf}')

# Display the first 5 rows of the transformed TF-IDF matrix
tfidf_sample = pd.DataFrame(title_features[:5], columns=[f'title_{i}' for i in range(title_features.shape[1])])
print(tfidf_sample)

# Get top 10 important words
feature_names = tfidf.get_feature_names_out()
print("Top 10 words in TF-IDF:", feature_names[:10])

sample_index = 0  # Choose a row to check
feature_array = title_features[sample_index]  # Get TF-IDF values for one row
word_importance = {word: feature_array[i] for i, word in enumerate(feature_names)}
sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most important words for this product:", sorted_words)

# Convert TF-IDF transformed data to a DataFrame
tfidf_df = pd.DataFrame(title_features, columns=[f'title_{i}' for i in range(title_features.shape[1])])

# Save as CSV
tfidf_df.to_csv("tfidf_transformed_data.csv", index=False)
print("TF-IDF data saved as CSV!")

# Model performance metrics
models = ['LightGBM', 'Ridge Regression']
mae_values = [mae_lgb, mae_ridge]
mse_values = [mse_lgb, mse_ridge]
r2_values = [r2_lgb, r2_ridge]

# Set up the bar chart
x = np.arange(len(models))  # X-axis positions
width = 0.3  # Bar width

fig, ax = plt.subplots(figsize=(10, 5))

# Plot bars
bar1 = ax.bar(x - width, mae_values, width, label='MAE', color='skyblue')
bar2 = ax.bar(x, mse_values, width, label='MSE', color='salmon')
bar3 = ax.bar(x + width, r2_values, width, label='R² Score', color='lightgreen')

# Labels and title
ax.set_xlabel("Models")
ax.set_ylabel("Performance Metrics")
ax.set_title("LightGBM vs Ridge Regression Performance")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Display the chart
plt.savefig("model_comparison_chart.png", dpi=1500)  # Save for PPT
plt.show()

# This will show the skewness of sold values before and after applying log transformation.

plt.figure(figsize=(12, 5))

# Original Sales Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['sold'], bins=50, kde=True)
plt.title('Original Sold Distribution')
plt.xlabel('Sold Items')
plt.ylabel('Frequency')

# Log-transformed Sales Distribution
plt.subplot(1, 2, 2)
sns.histplot(df['log_sold'], bins=50, kde=True, color='red')
plt.title('Log-Transformed Sold Distribution')
plt.xlabel('Log(Sold Items)')
plt.ylabel('Frequency')

plt.savefig("Log-Transformed Sold Distribution.png", dpi = 1500 , bbox_inches='tight')

plt.tight_layout()
plt.show()

# Relationship Between Price, Discount, and Sales

plt.figure(figsize=(8, 6))
scatter = plt.scatter(df['price'], df['sold'], c=df['discount'], cmap="coolwarm", alpha=0.5)

plt.colorbar(scatter, label="Discount (%)")  # Explicitly create a colorbar
plt.title("Effect of Price & Discount on Sales")
plt.xlabel("Price")
plt.ylabel("Sold Items")

plt.savefig("Effect of Price & Discount on Sales.png", dpi = 1500 , bbox_inches='tight')

plt.show()









































































































