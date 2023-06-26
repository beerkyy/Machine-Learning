import numpy as np
import matplotlib.pyplot as plt

#first we have to train the set

x_train = np.array([1.0, 2.0, 3.0, 4.0, 2.1, 3.4, 5.6, 4.6, 3.2, 5.6])             #size in 1000 square feet
y_train = np.array([300.0, 500.0, 400.0, 324.5, 234.5, 345.6, 755.4, 354.5, 325.4, 564.4])         #price in 1000s of dollars

#now we have to compute the cost
"""
The term 'cost' in this assignment might be a little confusing since the data is housing cost. Here, cost is a measure how well our model is predicting the target price of the house. The term 'price' is used for housing data.

The equation for cost with one variable is:
𝐽(𝑤,𝑏)=12𝑚∑𝑖=0𝑚−1(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2(1)
where
𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏(2)
𝑓𝑤,𝑏(𝑥(𝑖))
  is our prediction for example  𝑖
  using parameters  𝑤,𝑏
 .
(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))2
  is the squared difference between the target value and the prediction.
These differences are summed over all the  𝑚
  examples and divided by 2m to produce the cost,  𝐽(𝑤,𝑏)
 .
Note, in lecture summation ranges are typically from 1 to m, while code will be from 0 to m-1.

The code below calculates cost by looping over each example. In each loop:

f_wb, a prediction is calculated
the difference between the target and the prediction is calculated and squared.
this is added to the total cost.

"""
def temp_funtion(x,w,b):
    """
          Computes the prediction of a linear model
          Args:
            x (ndarray (m,)): Data, m examples
            w,b (scalar)    : model parameters
          Returns
            y (ndarray (m,)): target values
          """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

def compute_cost(x,y,w,b):
    """
        Computes the cost function for linear regression.

        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          w,b (scalar)    : model parameters

        Returns
            total_cost (float): The cost of using w,b as the parameters for linear regression
                   to fit the data points in x and y
        """
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb-y[i])**2
        cost_sum = cost_sum + cost
    total_cost = (1/(2*m))*cost_sum

    return total_cost

#cost function intuition

w_input = int(input("Please write the w value:"))
b_input = int(input("Pease write the b value:"))

tmp_cost_sum = compute_cost(x_train, y_train, w_input, b_input)
temp_f_wb = temp_funtion(x_train, w_input, b_input)

#plotting our model prediction and cost prediction model

plt.plot(x_train, temp_f_wb, c='b', label='Our Prediction')
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

plt.title("Housing Prices")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sqft)')
plt.legend()

#now lets create cost prediction model

plt.plot(w_input, compute_cost(x_train, y_train, w_input, b_input), c='b', label='Our Prediction')
plt.title("Cost Prediction Model")
# Set the y-axis label
plt.ylabel('J(w,b)')
# Set the x-axis label
plt.xlabel('w')
plt.legend()
plt.show()






