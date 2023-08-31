# Linear Regression with one feature
# def compute_cost(x, y, w, b):
# def compute_gradient(x, y, w, b):
# def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):

# model function: 𝑓𝑤,𝑏(𝑥)=𝑤𝑥+𝑏
# cost function: 𝐽(𝑤,𝑏)=1/2𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))^2
# gradient descent:
# 𝑏=𝑏−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑏
# ∂𝐽(𝑤,𝑏)/∂𝑏=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))
# 𝑤=𝑤−𝛼 ∂𝐽(𝑤,𝑏)/∂𝑤
# ∂𝐽(𝑤,𝑏)/∂𝑤=1𝑚∑(𝑓𝑤,𝑏(𝑥(𝑖))−𝑦(𝑖))𝑥(𝑖)

import numpy as np
import matplotlib.pyplot as plt

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
# show the plot
# plt.show()


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.
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