# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. numpy (np): numerical arrays and vectorized math.
2. pandas (pd): reading CSV and manipulating tabular data.
3. StandardScaler: standardizes features to zero mean and unit variance (useful for gradient descent stability).
4. 
## Program:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load dataset
data = pd.read_csv("exp_3_50_Startups.csv")

# Display first few rows
print("First 5 rows of the dataset:")
display(data.head())
# One-hot encode the 'State' column
data_encoded = pd.get_dummies(data, columns=['State'], drop_first=True)

# Display the encoded dataset
print("Encoded dataset:")
display(data_encoded.head())
# Independent features (all columns except Profit)
X_raw = data_encoded.drop('Profit', axis=1).values

# Target variable (Profit)
y_raw = data_encoded['Profit'].values.reshape(-1, 1)

print("Shape of features:", X_raw.shape)
print("Shape of target:", y_raw.shape)
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X = X_scaler.fit_transform(X_raw)
y = y_scaler.fit_transform(y_raw)
# Add a column of ones to X for bias term
m = X.shape[0]
X = np.hstack((np.ones((m, 1)), X))
print("Shape of X after adding bias term:", X.shape)
def compute_cost(X, y, theta):
    """
    Compute Mean Squared Error cost.
    """
    m = len(y)
    preds = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((preds - y) ** 2)
    return cost
def gradient_descent(X, y, learning_rate=0.01, num_iters=2000, tol=1e-8, verbose=False):
    """
    Batch Gradient Descent for Linear Regression.
    """
    m, n = X.shape
    theta = np.zeros((n, 1))
    J_history = []

    prev_cost = compute_cost(X, y, theta)

    for i in range(num_iters):
        preds = X.dot(theta)
        errors = preds - y
        grad = (1 / m) * (X.T.dot(errors))
        theta -= learning_rate * grad

        cost = compute_cost(X, y, theta)
        J_history.append(cost)

        if abs(prev_cost - cost) < tol:
            if verbose:
                print(f"Converged at iteration {i}")
            break
        prev_cost = cost

        if verbose and (i % 500 == 0 or i < 5):
            print(f"Iteration {i:4d}, Cost: {cost:.6f}")

    return theta, J_history
alpha = 0.01
theta, J_hist = gradient_descent(X, y, learning_rate=alpha, num_iters=5000, tol=1e-9, verbose=True)

print("\nLearned Parameters (Theta):")
print(theta.flatten())
plt.figure(figsize=(7,4))
plt.plot(J_hist)
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE/2)')
plt.title('Cost Function Convergence')
plt.grid(True)
plt.show()
```
```python
# Create new sample (same feature names as original)
new_sample = pd.DataFrame([{
    'R&D Spend': 165349.2,
    'Administration': 136897.8,
    'Marketing Spend': 471784.1,
    'State': 'New York'
}])

# Apply one-hot encoding
new_encoded = pd.get_dummies(new_sample, columns=['State'], drop_first=True)

# Align columns with training data (add missing ones as 0)
new_encoded = new_encoded.reindex(columns=data_encoded.drop('Profit', axis=1).columns, fill_value=0)

# Scale new data
new_scaled = X_scaler.transform(new_encoded)

# Add bias
new_design = np.hstack((np.ones((new_scaled.shape[0], 1)), new_scaled))

# Predict (scaled)
scaled_pred = new_design.dot(theta)

# Inverse transform to original units
pred_original = y_scaler.inverse_transform(scaled_pred)

print(f"\nPredicted Profit for the new startup: ₹{pred_original[0][0]:,.2f}")


/*
Program to implement the linear regression using gradient descent.
Developed by: karthik J
RegisterNumber: 212225040176
*/
```

## Output:

<img width="820" height="533" alt="Screenshot 2026-07-27 132035" src="https://github.com/user-attachments/assets/0cf5508a-db33-474a-9e42-b5ac5fc8bf44" />
<img width="798" height="420" alt="Screenshot 2026-07-27 132051" src="https://github.com/user-attachments/assets/c01ec471-31fd-4f0c-93a6-57825b29bab0" />
<img width="885" height="495" alt="Screenshot 2026-07-27 132101" src="https://github.com/user-attachments/assets/d5661031-e0b7-4a26-9346-6c722e2881e3" />
<img width="1252" height="141" alt="Screenshot 2026-07-27 132128" src="https://github.com/user-attachments/assets/3c1644fb-806b-49ed-948a-58ffccb34edc" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
