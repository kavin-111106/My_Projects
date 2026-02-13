
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
print("Loading Dataset...")
df = pd.read_csv("data/train.csv") 
print("Dataset Loaded Successfully \n")

features = ['GrLivArea', 'FullBath', 'BedroomAbvGr']
target = 'SalePrice'
data = df[features + [target]]
print("Checking for missing values...")
print(data.isnull().sum())
data.dropna(inplace=True)
print("Missing values removed \n")

X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training Linear Regression Model...")
model = LinearRegression()
model.fit(X_train, y_train)
print("Model Trained \n")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}\n")

coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
})

print("Feature Coefficients:")
print(coefficients)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices (Linear Regression)")
plt.grid(True)
plt.tight_layout()
plt.savefig("price_prediction_plot.png")   
plt.show()

print("\nGraph Saved as 'price_prediction_plot.png' ")
