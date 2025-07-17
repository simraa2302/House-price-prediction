# house_price_prediction.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load Kaggle dataset
data = pd.read_csv("kc_house_data.csv")

# Step 2: Use only required features
X = data[["bedrooms", "sqft_living"]]
y = data["price"]

# Optional: Preview data
print("Sample Data:\n", data[["bedrooms", "sqft_living", "price"]].head())

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Show coefficients
print("\nModel Intercept:", model.intercept_)
print("Model Coefficients:", model.coef_)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

# Step 8: Visualize predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Kaggle Dataset: Actual vs Predicted House Prices")
plt.grid(True)
plt.show()
