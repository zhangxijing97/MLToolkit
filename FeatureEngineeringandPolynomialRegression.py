# Feature engineering and Polynomial regression
##  Dataset:
# | Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |
# | ----------------| ------------------- |----------------- |--------------|----------------------- |
# | 952             | 2                   | 1                | 65           | 271.5                  |
# | 1244            | 3                   | 2                | 64           | 232                    |
# | 1947            | 3                   | 2                | 17           | 509.8                  |
# | ...             | ...                 | ...              | ...          | ...                    |

import numpy as np
import matplotlib.pyplot as plt
from MultipleLinearRegression import gradient_descent

# load data
def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    X = data[:,:4]
    y = data[:,4]
    return X, y

# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

# view the dataset and its features by plotting
fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()