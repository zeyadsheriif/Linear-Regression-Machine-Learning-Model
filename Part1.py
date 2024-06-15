!pip install -U scikit-learn
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define the linear equation (without noise)
def linear_eq(x):
    return 3 * x + 2

# Generate x values
x_values = np.array([1, 2, 3, 4, 5])

# Calculate y values for both equations
y_values = linear_eq(x_values)

# Plot the data points
plt.scatter(x_values, y_values, label='Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points')
plt.legend()
plt.show()

# Reshape the data for sklearn
x_values_sklearn = x_values.reshape(-1, 1)
y_values_sklearn = y_values.reshape(-1, 1)

# Perform linear regression
regression = LinearRegression().fit(x_values_sklearn, y_values_sklearn)

# Get the coefficients
slope = regression.coef_[0][0]
intercept = regression.intercept_[0]

# Print the equations of the lines
print("Equation of the line without noise:")
print(f"y = {slope}x + {intercept}")

# Define the linear equation with random noise
def linear_eq_with_noise(x):
    return 3 * x + 2 + np.random.normal(0, 0.1, len(x))

# Generate x values
x_values = np.array([1, 2, 3, 4, 5])

# Calculate y values for both equations
y_values_with_noise = linear_eq_with_noise(x_values)

# Plot the data points
plt.scatter(x_values, y_values_with_noise, label='Data with Noise')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points')
plt.legend()
plt.show()

# Reshape the data for sklearn
y_values_with_noise_sklearn = y_values_with_noise.reshape(-1, 1)

# Perform linear regression
regression_with_noise = LinearRegression().fit(x_values_sklearn, y_values_with_noise_sklearn)

slope_with_noise = regression_with_noise.coef_[0][0]
intercept_with_noise = regression_with_noise.intercept_[0]

print("\nEquation of the line with noise:")
print(f"y = {slope_with_noise}x + {intercept_with_noise}")

# Plot the data points
plt.scatter(x_values, y_values, label='Original Data')
plt.scatter(x_values, y_values_with_noise, label='Data with Noise')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Data Points')
plt.legend()
plt.show()
