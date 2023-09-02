# Multiple linear regression
# 1. def predict(x, w, b):
# 2. def compute_cost(x, y, w, b):
# 3. def compute_gradient(x, y, w, b):
# 4. def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

# model function: 𝑓𝑤,𝑏(𝑥)=𝑤𝑥+𝑏
# cost function: 𝐽(𝑤,𝑏)=1/2𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))^2
# gradient descent:
# 𝑏=𝑏−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑏
# ∂𝐽(𝑤,𝑏)/∂𝑏=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))
# 𝑤=𝑤−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑤
# ∂𝐽(𝑤,𝑏)/∂𝑤=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖)

import copy, math
import numpy as np
import matplotlib.pyplot as plt

# Problem Statement
# Size (sqft)     Number of Bedrooms     Number of floors     Age of Home     Price (1000s dollars)
# 2104	          5	                     1	                  45	          460
# 1416	          3	                     2	                  40	          232
# 852	          2	                     1	                  35	          178
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")

# 1. predict function
def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")
# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

# 2. compute_cost
def compute_cost(X, y, w, b):
    """
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b  # (n,)(n,) = scalar (see np.dot)
        cost = cost + (f_wb_i - y[i]) ** 2  # scalar
    cost = cost / (2 * m)  # scalar
    return cost

# 2. compute_gradient
# ∂𝐽(𝑤,𝑏)/∂𝑏 is dj_db
# gradient = derivative = rate of change of a function
# When w,b is a certain value, find the derivative of the cost function
def compute_gradient(x, y, w, b):
    """
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
    Returns
      dj_dw (scalar): The gradient of the cost with respect to the parameters w
      dj_db (scalar): The gradient of the cost with respect to the parameter b
     """
    # gradient descent:
    # 𝑏=𝑏−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑏
    # ∂𝐽(𝑤,𝑏)/∂𝑏=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))
    # 𝑤=𝑤−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑤
    # ∂𝐽(𝑤,𝑏)/∂𝑤=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖)

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i  # update derivative of ∂𝐽(𝑤,𝑏)/∂b
        dj_dw += dj_dw_i  # update derivative of ∂𝐽(𝑤,𝑏)/∂w
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db

# Compute and display cost using our pre-chosen optimal parameters.
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')
# Expected Result: Cost at optimal w : 1.5578904045996674e-12

# 3. compute_gradient
# ∂𝐽(𝑤,𝑏)/∂𝑏 is dj_db
# gradient = derivative = rate of change of a function
# When w,b is a certain value, find the derivative of the cost function
def compute_gradient(x, y, w, b):
    """
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
    Returns
      dj_dw (scalar): The gradient of the cost with respect to the parameters w
      dj_db (scalar): The gradient of the cost with respect to the parameter b
     """
    # gradient descent:
    # 𝑏=𝑏−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑏
    # ∂𝐽(𝑤,𝑏)/∂𝑏=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))
    # 𝑤=𝑤−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑤
    # ∂𝐽(𝑤,𝑏)/∂𝑤=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖)

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i  # update derivative of ∂𝐽(𝑤,𝑏)/∂b
        dj_dw += dj_dw_i  # update derivative of ∂𝐽(𝑤,𝑏)/∂w
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db


def compute_gradient(X, y, w, b):
    """
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b.
    """


    return dj_db, dj_dw

