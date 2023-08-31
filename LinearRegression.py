# Linear Regression with one feature
import numpy as np
import matplotlib.pyplot as plt

# load data
def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:,0] # all the first column of the dataset
    y = data[:,1] # all the second colum of the dataset
    return X, y

def load_data_multi():
    data = np.loadtxt("data/ex1data2.txt", delimiter=',')
    X = data[:,:2] 
    y = data[:,2] # all the third column of the dataset
    return X, y

x_train, y_train = load_data()

# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5])

# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])