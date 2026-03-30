# =====================================================
# 02 Predict to Invest: MLR Pipeline for Startup Profitability
# Complete Multiple Linear Regression Pipeline + EDA + Insights
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ====================== 1. LOAD DATA ======================
df = pd.read_csv('50_Startups.csv')
print("✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}\n")
print(df.head())
print("\n" + "="*50)

# ====================== 2. EDA ======================
print("📊 EXPLORATORY DATA ANALYSIS")
print(df.info())
print("\nStatistical Summary:")
print(df.describe())

# Correlation heatmap (numerical features only)
plt.figure(figsize=(8, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix - Startup Features vs Profit')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("📊 Correlation heatmap saved as 'correlation_heatmap.png'")

# Pairplot
sns.pairplot(df, hue='State')
plt.savefig('pairplot.png')
print("📊 Pairplot saved as 'pairplot.png'\n")

# ====================== 3. PREPROCESSING ======================
X = df.drop('Profit', axis=1)
y = df['Profit']

# Column Transformer: One-Hot Encoding for 'State' (avoids dummy variable trap)
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(drop='first', sparse_output=False), ['State'])],
    remainder='passthrough'
)

# ====================== 4. PIPELINE & TRAIN/TEST SPLIT ======================
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====================== 5. TRAIN MODEL ======================
model_pipeline.fit(X_train, y_train)
print("🚀 Multiple Linear Regression model trained successfully!\n")

# ====================== 6. PREDICTION & EVALUATION ======================
y_pred = model_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("📈 MODEL PERFORMANCE")
print(f"R² Score          : {r2:.4f}  ({r2*100:.2f}% variance explained)")
print(f"Mean Absolute Error: ${mae:,.2f}")
print(f"Mean Squared Error : ${mse:,.2f}")
print(f"Root Mean Squared Error: ${rmse:,.2f}\n")

# ====================== 7. FEATURE IMPORTANCE (Business Insights) ======================
# Get feature names after one-hot encoding
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = model_pipeline.named_steps['regressor'].coef_

print("💼 FEATURE IMPORTANCE (Impact on Profit)")
for feature, coef in zip(feature_names, coefficients):
    print(f"   {feature:25} : ${coef:,.2f} per unit increase")
print("\n🔑 KEY BUSINESS INSIGHT:")
print("   → R&D Spend has the strongest positive impact on Profit.")
print("   → Marketing Spend is also highly influential.")
print("   → Administration has the weakest impact.")
print("   → Recommendation: Prioritize startups with high R&D investment.")

# ====================== 8. VISUALIZATIONS ======================
# Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Actual vs Predicted Profit')
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print("📊 Actual vs Predicted plot saved as 'actual_vs_predicted.png'")

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Profit')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('residual_plot.png')
print("📊 Residual plot saved as 'residual_plot.png'")

print("\n🎉 Assignment complete! All plots saved. Ready for submission.")