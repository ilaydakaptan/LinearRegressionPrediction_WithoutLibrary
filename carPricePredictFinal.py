import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class LinearRegression:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.cost_values = []
        self.theta = np.ones(6)

    def cost_func(self, X, y):
        m = len(y)
        h = np.dot(X, self.theta)
        errors = np.power((h - y), 2)
        J = 1 / (2 * m) * np.sum(errors)
        self.cost_values.append(J)
        return J

    def theta_func(self, X, y):
        X = X.values
        m = len(y)
        h = np.dot(X, self.theta)
        errors = h - y

        for i in range(len(self.theta)):
            gradient = np.sum(errors * X[:, i]) / m
            self.theta[i] -= self.learning_rate * gradient

    def gradient_descent(self, X, y):
        before_iteration = 1
        later_iteration = 0
        counter = 1
        while before_iteration > later_iteration and before_iteration - later_iteration > self.learning_rate:
            before_iteration = self.cost_func(X, y)
            self.theta_func(X, y)
            later_iteration = self.cost_func(X, y)
            counter += 1
        return counter

    def fit(self, X, y):
        return self.gradient_descent(X, y)

    def predict(self, X_test):
        return np.dot(X_test, self.theta)

    def evaluate(self, predictions, y_test):
        mse = mean_squared_error(predictions, y_test)
        mae = mean_absolute_error(predictions, y_test)
        r_squared = r2_score(predictions, y_test)
        print("R Squared Error (r2):", r_squared)
        print("Mean Squared Error (MSE): ", mse)
        print("Mean Absolute Error (MAE):", mae)

    def cost_history(self):
        iteration = list(range(0, len(self.cost_values)))
        plt.plot(iteration, self.cost_values)
        plt.title("Learning Rate=" + str(self.learning_rate))
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()

# Read and preprocess data
test_data = pd.read_csv("testDATA.csv")
train_data = pd.read_csv("trainDATA.csv")

train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()

# Encoding the categorical feautures
encoder = LabelEncoder()
train_data["fuel"] = encoder.fit_transform(train_data["fuel"])
train_data["seller_type"] = encoder.fit_transform(train_data["seller_type"])
train_data["transmission"] = encoder.fit_transform(train_data["transmission"])
train_data["owner"] = encoder.fit_transform(train_data["owner"])

test_data["fuel"] = encoder.fit_transform(test_data["fuel"])
test_data["seller_type"] = encoder.fit_transform(test_data["seller_type"])
test_data["transmission"] = encoder.fit_transform(test_data["transmission"])
test_data["owner"] = encoder.fit_transform(test_data["owner"])

# Scaling the numeric features
scaler = MinMaxScaler()
train_data["year"] = scaler.fit_transform(train_data["year"].values.reshape(-1, 1))
train_data["km_driven"] = scaler.fit_transform(train_data["km_driven"].values.reshape(-1, 1))
train_data["selling_price"] = scaler.fit_transform(train_data["selling_price"].values.reshape(-1, 1))

test_data["year"] = scaler.fit_transform(test_data["year"].values.reshape(-1, 1))
test_data["km_driven"] = scaler.fit_transform(test_data["km_driven"].values.reshape(-1, 1))
test_data["selling_price"] = scaler.fit_transform(test_data["selling_price"].values.reshape(-1, 1))

X = train_data[["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
Y = train_data["selling_price"]

# Try different learning rates
learning_rates = [0.001, 0.003, 0.005, 0.008, 0.01]
best_learning_rate = None
best_cost = float('inf')

for lr in learning_rates:
    linear_regression_model = LinearRegression(learning_rate=lr)
    iterations = linear_regression_model.fit(X, Y)
    print(f"Learning Rate: {lr}, Iterations: {iterations}, Final Cost: {linear_regression_model.cost_values[-1]}")

    if linear_regression_model.cost_values[-1] < best_cost:
        best_cost = linear_regression_model.cost_values[-1]
        best_learning_rate = lr

print(f"Best Learning Rate: {best_learning_rate}, Best Cost: {best_cost}")

# Use the best learning rate for final training
linear_regression_model = LinearRegression(learning_rate=best_learning_rate)
linear_regression_model.fit(X, Y)

# Make predictions on the test data
X_test = test_data[["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
predictions = linear_regression_model.predict(X_test)

# Evaluate the model on the test data
Y_test = test_data["selling_price"]
linear_regression_model.evaluate(predictions, Y_test)
linear_regression_model.cost_history()

# Create a DataFrame with the predictions
results_df = pd.DataFrame({"Actual": Y_test, "Predicted": predictions})
results_df.to_csv("predictions.csv", index=False)

