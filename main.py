#import the NumPy and Matplotlib

import numpy as np
import matplotlib.pyplot as plt

#define the x_train and y_train

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0,500.0])

print(f"x_train: {x_train}")
print(f"y_train : {y_train}")

#number of training examples

"""
m is the number of training examples

"""
m = x_train.shape[0]
print(f"Number of training examples : {m}")

#looking for the training examples

i = 0
x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

#graphing the actual outputs
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()

#we are building linear regression model so the f function will be f = w * x[i] + b
#lets first define the values of w and b

w = 100
b = 100

#then we have to calculate the function thats why we are going to use a function

def compute_model_output(x,b,w):
    """
      Computes the prediction of a linear model
      Args:
        x (ndarray (m,)): Data, m examples
        w,b (scalar)    : model parameters
      Returns
        y (ndarray (m,)): target values
      """
    m = x.shape[0]
    #we have to create zero array
    f_wb = np.zeros(m)
    for i in range (m):
        f_wb[i] = w * x[i] + b
    return f_wb
tmp_f_wb = compute_model_output(x_train,b,w)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()

#calculate the cost
w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")