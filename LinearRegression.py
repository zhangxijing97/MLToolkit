# Linear Regression with one feature
# 1. def compute_cost(x, y, w, b):
# 2. def compute_gradient(x, y, w, b):
# 3. def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

# model function: 𝑓𝑤,𝑏(𝑥)=𝑤𝑥+𝑏
# cost function: 𝐽(𝑤,𝑏)=1/2𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))^2
# gradient descent:
# 𝑏=𝑏−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑏
# ∂𝐽(𝑤,𝑏)/∂𝑏=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))
# 𝑤=𝑤−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑤
# ∂𝐽(𝑤,𝑏)/∂𝑤=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖)

import numpy as np
import matplotlib.pyplot as plt
import copy # a method that is used on objects to create copies of them
import math

# load data
def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:,0] # all the first column of the dataset
    y = data[:,1] # all the second column of the dataset
    return X, y

def load_data_multi():
    data = np.loadtxt("data/ex1data2.txt", delimiter=',')
    X = data[:,:2] # take all rows and all but the third column
    y = data[:,2] # all the third column of the dataset
    return X, y

x_train, y_train = load_data()

# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])

# check the dimensions of your variables
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))

# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

# 1. compute_cost
def compute_cost(x, y, w, b):
    """
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
    """
    # cost function: 𝐽(𝑤,𝑏)=1/2𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))^2
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1 / (2 * m) * cost
    return total_cost

# Compute cost with some initial values for paramaters w, b
initial_w = 2
initial_b = 1
cost = compute_cost(x_train, y_train, initial_w, initial_b)
print(type(cost))
print(f'Cost at initial w: {cost:.3f}')
# Expected Output: Cost at initial w: 75.203

# 2. compute_gradient
# ∂𝐽(𝑤,𝑏)/∂𝑏 is dj_db
# gradient = derivative = rate of change of a function
# When w,b is a certain value, find the derivative of the cost function
def compute_gradient(x, y, w, b):
    """
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
    Returns
      dj_dw (scalar): The gradient of the cost with respect to the parameters w
      dj_db (scalar): The gradient of the cost with respect to the parameter b
     """
    # gradient descent:
    # 𝑏=𝑏−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑏
    # ∂𝐽(𝑤,𝑏)/∂𝑏=1/𝑚 * ∑(𝑓𝑤,𝑏(X(𝑖))−𝑦(𝑖))
    # 𝑤=𝑤−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑤
    # ∂𝐽(𝑤,𝑏)/∂𝑤=1/𝑚 * ∑(𝑓𝑤,𝑏(X(𝑖))−𝑦(𝑖)) * 𝑥(𝑖)

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

# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)
# Expected Output: Gradient at initial w, b (zeros): -65.32884974555672 -5.83913505154639

test_w = 0.2
test_b = 0.2
tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, test_w, test_b)
print('Gradient at test w, b:', tmp_dj_dw, tmp_dj_db)
# Expected Output: Gradient at test w, b: -47.41610118114435 -4.007175051546391

# 3. gradient_descent
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
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
    b = b_in
    w = w_in
    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)  # get ∂𝐽(𝑤,𝑏)/∂w and ∂𝐽(𝑤,𝑏)/∂b
        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))  # store all total_cost
            p_history.append([w, b])  # store all w,b
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
    return w, b, J_history, p_history  # return w and J,w history for graphing
# Note: math.ceil() round up the result

# initialize fitting parameters. Recall that the shape of w is (n,)
initial_w = 0.
initial_b = 0.
# some gradient descent settings
iterations = 1500
alpha = 0.01
w,b,J_history,p_history = gradient_descent(x_train ,y_train, initial_w, initial_b, alpha, iterations, compute_cost, compute_gradient)
print("w,b found by gradient descent:", w, b)
# Expected Output: w, b found by gradient descent 1.16636235 -3.63029143940436

m = x_train.shape[0]
predicted = np.zeros(m)
for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot the linear fit
plt.plot(x_train, predicted, c = "b")
# Create a scatter plot of the data.
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')

# population test
predict1 = 3.5 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))
predict2 = 7.0 * w + b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

# show the plot
# plt.show()

# scientific notation example
number = 123.123
formatted_number = f"{number:e}"  # transfer to scientific notation
print(formatted_number)
formatted_number = f"{number:0.2e}"  # .2e means two digits after the decimal point
print(formatted_number)