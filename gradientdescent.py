import math
import numpy as np
import matplotlib.pyplot as plt


#first we have to load our data set

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w_initial = int(input("Please write a desired initial w value:"))
b_initial = int(input("Please write a desired initial b value:"))
alpha_initial = float(input("Please write a desired initial alpha value:"))
iterations = int(input("Please write a desired interation [max : 100000] value:"))
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

#then we have to develop a cost return function
def compute_gradient(x,y,w,b):
    #to compute ∂𝐽(𝑤,𝑏)∂𝑤 and ∂𝐽(𝑤,𝑏)∂𝑏
    """
        Computes the gradient for linear regression
        Args:
          x (ndarray (m,)): Data, m examples
          y (ndarray (m,)): target values
          w,b (scalar)    : model parameters
        Returns
          dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
          dj_db (scalar): The gradient of the cost w.r.t. the parameter b
         """
    #firstly, we need to find the lenght of the training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb-y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_db += dj_db_i
        dj_dw += dj_dw_i

        #then we have to calculate
    dj_dw = dj_dw /m
    dj_db = dj_db /m
    return dj_dw,dj_db

def gradient_descent(x,y,w,b,alpha,num_iters,cost_function,gradient_function):

    """
       Performs gradient descent to fit w,b. Updates w,b by taking
       num_iters gradient steps with learning rate alpha

       Args:
         x (ndarray (m,))  : Data, m examples
         y (ndarray (m,))  : target values
         w_in,b_in (scalar): initial values of model parameters
         alpha (float):     Learning rate
         num_iters (int):   number of iterations to run gradient descent
         cost_function:     function to call to produce cost
         gradient_function: function to call to produce gradient

       Returns:
         w (scalar): Updated value of parameter after running gradient descent
         b (scalar): Updated value of parameter after running gradient descent
         J_history (List): History of cost values
         p_history (list): History of parameters [w,b]
         """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []

    for i in range(num_iters):
        #calculating the gradient and update the parameters using already coded function gradient_descent

        dj_dw,dj_db = gradient_function(x_train,y_train,w,b)

        #now we have to update the parameters

        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        #saving cost J for each iteration

        if i < 100000:
            J_history.append(cost_function(x,y,w,b))
            p_history.append([w,b])
        #print the interations and values

        if i%math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history


w_final, b_final, J_hist,p_hist = w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train,
                        w_initial, b_initial,alpha_initial,iterations,compute_cost,compute_gradient)

print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")





