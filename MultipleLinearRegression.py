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

# 3. compute_gradient
# ∂𝐽(𝑤,𝑏)/∂𝑏 is dj_db
# gradient = derivative = rate of change of a function
# When w,b is a certain value, find the derivative of the cost function
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
    # one feature:
    # 𝑏 = 𝑏−𝛼 ∂𝐽(𝑤, 𝑏) /∂𝑏
    # ∂𝐽(𝑤, 𝑏) /∂𝑏 = 1 / 𝑚 * ∑(𝑓𝑤, 𝑏(X(𝑖))−𝑦(𝑖))
    # 𝑤 = 𝑤−𝛼 ∂𝐽(𝑤, 𝑏) /∂𝑤
    # ∂𝐽(𝑤, 𝑏) /∂𝑤 = 1 / 𝑚 * ∑(𝑓𝑤, 𝑏(X(𝑖))−𝑦(𝑖)) * 𝑥(𝑖)

    # n features:
    # 𝑏 = 𝑏−𝛼 ∂𝐽(𝑤, 𝑏) /∂𝑏
    # ∂𝐽(𝑤, 𝑏) /∂𝑏 = 1 / 𝑚 * ∑(𝑓𝑤, 𝑏(X(𝑖))−𝑦(𝑖))
    # 𝑤1 = 𝑤1−𝛼 * 1 / 𝑚 * ∑(𝑓𝑤, 𝑏(X(𝑖))−𝑦(𝑖)) * 𝑥(𝑖)1
    # 𝑤2 = 𝑤2−𝛼 * 1 / 𝑚 * ∑(𝑓𝑤, 𝑏(X(𝑖))−𝑦(𝑖)) * 𝑥(𝑖)2
    # note:
    # xj = jth feature
    # x(i) = ith example
    # x(i)j = jth feature of ith example

    # m(3) rows/example,
    # n(4) columns/features
    m,n = X.shape
    dj_dw = np.zeros((n,)) # get 4*1(rows 1 columns) matrix
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i] # calculate the err of y and y hat
        for j in range(n): # get w1, w2, wj ... for all x
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

#Compute and display gradient
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')
# Expected Result:
# dj_db at initial w,b: -1.6739251122999121e-06
# dj_dw at initial w,b: [-2.73e-03 -6.27e-06 -2.22e-06 -6.92e-05]

# 4. def gradient_descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
    Returns:
      w (ndarray (n,)) : Updated values of parameters
      b (scalar)       : Updated value of parameter
      """
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)  ##None
        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw  ##None
        b = b - alpha * dj_db  ##None
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(X, y, w, b))
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history  # return final w,b and J history for graphing

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
# Expected Result:
# b,w found by gradient descent: -0.00,[ 0.2 0. -0.01 -0.07]
# prediction: 426.19, target value: 460
# prediction: 286.17, target value: 232
# prediction: 171.47, target value: 178

# plot cost versus iteration
# fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
# ax1.plot(J_hist)
# ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
# ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
# ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost')
# ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step')
