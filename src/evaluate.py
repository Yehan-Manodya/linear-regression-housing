import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from train import train_model

# Train model and get test data
model, X_test, y_test = train_model()

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Create folder 'plots' if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")

# Save plot to file
plt.savefig("plots/actual_vs_predicted.png")  # ✅ saves the image
plt.show()
