#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import PolynomialFeatures
data = pd.read_csv("insurance.txt")


# ## Class for Models

# In[2]:


class Models:
    def __init__(self, data):
        self.data = data.drop(columns = ["children"]).sample(frac=1)
        self.GDModels = pd.DataFrame(columns = ["training_error", "testing_error", "validation_error", "degree", "l1", "l2", "iterations"])
        self.SGDModels = pd.DataFrame(columns = ["training_error", "testing_error", "validation_error", "degree", "l1", "l2", "iterations"])
  
    def extractFeatures(self):
        x1 = self.data["age"]
        x1 = np.array(x1)
        x1 = x1.reshape((len(x1), 1))
        x2 = self.data["bmi"]
        x2 = np.array(x2)
        x2 = x2.reshape((len(x2), 1))
        Y = self.data["charges"]
        Y = np.array(Y)
        Y = Y.reshape((len(Y), 1))
        return np.concatenate((x1, x2), axis=1), Y

    def polyFeatures(self, degree, X):
        poly = PolynomialFeatures(degree)
        return poly.fit_transform(X)
  
    def normalize(self, X):
        norm = lambda x: ((x - x.min()) / (x.max() - x.min()))
        return norm(X)
  
    def split(self, X):
        X_train = X[:int(0.7 * len(X))]
        X_val = X[int(0.7 * len(X)):int(0.9 * len(X))]
        X_test = X[int(0.9 * len(X)):]
        return X_train, X_val, X_test
  
    # Gradient Descent Algorithm
    def GD(self, X, X_val, Y, Y_val, l1 = 0, l2 = 0, learning_rate = 1e-5):
        sgn = lambda x: (x / abs(x)) #signum function
        W = np.random.randn(X.shape[1])
        W = W.reshape((X.shape[1], 1))
        prev_error, error = float("0"), float("inf")
        count = 0
        while abs(prev_error - error) >= 1e-5 and count < 25000 - 1:
            Y_pred = X @ W
            hypothesis = (Y_pred - Y)
            prev_error = error
            error = 0.5 * (np.sum(np.square(hypothesis)) + l1 * np.sum(np.abs(W)) + l2 * np.sum(np.square(W)))
            sgn_w = np.array([sgn(w) for w in W]).reshape((X.shape[1], 1))
            W -= learning_rate * ((X.T @ hypothesis) + 0.5 * l1 * sgn_w + l2 * W)
            count += 1
            if count % 5000 == 0:
                print(f"Epochs = {count} || Error = {error} || L1 = {l1} || L2 = {l2}")
        val_D = ((X_val @ W) - Y_val)
        val_loss = 0.5 * (np.sum(np.square(val_D)) + l1 * np.sum(np.abs(W)) + l2 * np.sum(np.square(W)))
        return error, val_loss, W, count + 1
  
    def SGD(self, X, X_val, Y, Y_val, l1 = 0, l2 = 0, learning_rate = 1e-7):
        sgn = lambda x: (x / abs(x)) #signum function
        W = np.random.randn(X.shape[1])
        W = W.reshape((X.shape[1], 1))
        prev_error, error = float("0"), float("inf")
        count = 0
        while abs(prev_error - error) >= 1e-6 and count < 25000 - 1:
            x = X[count%X.shape[0]].reshape(1, X.shape[1])
            y = Y[count%X.shape[0]].reshape(1, 1)
            hypo = ((X @ W) - Y)
            hypothesis = ((x @ W) - y)
            prev_error = error
            error = 0.5 * (np.sum(np.square(hypo)) + l1 * np.sum(np.abs(W)) + l2 * np.sum(np.square(W)))
            sgn_w = np.array([sgn(w) for w in W]).reshape((X.shape[1], 1))
            W -= learning_rate * ((x.T @ hypothesis) + 0.5 * l1 * sgn_w + l2 * W)
            count += 1
            if count % 5000 == 0:
                print(f"Iterations = {count} || Error = {error} || L1 = {l1} || L2 = {l2}")
        val_D = ((X_val @ W) - Y_val)
        val_loss = 0.5 * (np.sum(np.square(val_D)) + l1 * np.sum(np.abs(W)) + l2 * np.sum(np.square(W)))
        return error, val_loss, W, count + 1
  
    def plot(self, X_feat, W, degree, method):
        x1 = X_feat[:,0]
        x2 = X_feat[:,1]
#         x1, x2 = np.meshgrid(x1, x2)
        X = self.polyFeatures(degree, X_feat)
        Y_pred = X @ W
        fig = plt.figure(figsize=(8,8))
        axes = fig.gca(projection='3d')
        axes.plot_trisurf(x1.flatten(), x2.flatten(), Y_pred.flatten(),cmap='viridis')
        axes.set_xlabel("X1")
        axes.set_ylabel("X2")
        axes.set_zlabel("Y pred")
        name = str(degree) + method
        axes.figure.savefig(f"{name}.jpg")
        plt.close()
  
    def buildModels(self, low = 1, high = 10):
        for degree in range(low, high + 1):
            X_feat, Y_feat = self.extractFeatures()
            X = self.polyFeatures(degree, X_feat)
            X = self.normalize(X)
            Y = self.normalize(Y_feat)
            X_train, X_val, X_test = self.split(X)
            Y_train, Y_val, Y_test = self.split(Y)
            print(f"Gradient Descent for Regression without Regularisation || model degree {degree}")
            training_error, val_loss, W , i = self.GD(X_train, X_val, Y_train, Y_val)
            test_D = ((X_test @ W) - Y_test)
            testing_error = error = 0.5 * (np.sum(np.square(test_D)))
            GDModel = [training_error, testing_error, val_loss, degree, 0, 0, i]
            GDModel = pd.Series(GDModel, index = self.GDModels.columns)
            self.GDModels = self.GDModels.append(GDModel, ignore_index=True)
            self.plot(X_feat, W, degree, "GD")
            # GD Lasso Regression
            print(f"Gradient Descent for Lasso Regression || model degree {degree}")
            L1 = list(np.random.rand(5))
            min_val_loss = float("inf")
            training_error_final = float("inf")
            W_final = W
            l1_final = 0
            i_final = 0
            for l1 in L1:
                training_error, val_loss, W, i = self.GD(X_train, X_val, Y_train, Y_val, l1=l1)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    W_final = W
                    training_error_final = training_error
                    l1_final = l1
                    i_final = i
            test_D = ((X_test @ W_final) - Y_test)
            testing_error_final = error = 0.5 * (np.sum(np.square(test_D)) + l1_final * np.sum(np.abs(W_final)))
            GDModel1 = [training_error_final, testing_error_final, min_val_loss, degree, l1_final, 0, i_final]
            GDModel1 = pd.Series(GDModel1, index = self.GDModels.columns)
            self.GDModels = self.GDModels.append(GDModel1, ignore_index=True)
            # GD Ridge Regression
            print(f"Gradient Descent for Ridge Regression || model degree {degree}")
            L2 = list(np.random.rand(5))
            min_val_loss = float("inf")
            training_error_final = float("inf")
            W_final = W
            l2_final = 0
            i_final = 0
            for l2 in L2:
                training_error, val_loss, W, i = self.GD(X_train, X_val, Y_train, Y_val, l2=l2)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    W_final = W
                    training_error_final = training_error
                    l2_final = l2
                    i_final = i
            test_D = ((X_test @ W_final) - Y_test)
            testing_error_final = error = 0.5 * (np.sum(np.square(test_D)) + l2_final * np.sum(np.square(W_final)))
            GDModel2 = [training_error_final, testing_error_final, min_val_loss, degree, 0, l2_final, i_final]
            GDModel2 = pd.Series(GDModel2, index = self.GDModels.columns)
            self.GDModels = self.GDModels.append(GDModel2, ignore_index=True)
            # Stochastic Gradient Descent
            print(f"Stochastic Gradient Descent for Regression without Regularisation || model degree {degree}")
            training_error, val_loss, W, i = self.SGD(X_train, X_val, Y_train, Y_val) # gradient descent without regularisation
            test_D = ((X_test @ W) - Y_test)
            testing_error = error = 0.5 * (np.sum(np.square(test_D)))
            SGDModel = [training_error, testing_error, val_loss, degree, 0, 0, i]
            SGDModel = pd.Series(SGDModel, index = self.SGDModels.columns)
            self.SGDModels = self.SGDModels.append(SGDModel, ignore_index=True)
            # self.plot(X, Y, W, degree, "SGD")
            # SGD Lasso Regression
            print(f"Stochastic Gradient Descent for Lasso Regression || model degree {degree}")
            L1 = list(np.random.rand(5))
            min_val_loss = float("inf")
            training_error_final = float("inf")
            W_final = W
            l1_final = 0
            i_final = 0
            for l1 in L1:
                training_error, val_loss, W, i = self.SGD(X_train, X_val, Y_train, Y_val, l1=l1)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    W_final = W
                    training_error_final = training_error
                    l1_final = l1
                    i_final = i
            test_D = ((X_test @ W_final) - Y_test)
            testing_error_final = error = 0.5 * (np.sum(np.square(test_D)) + l1_final * np.sum(np.abs(W_final)))
            SGDModel1 = [training_error_final, testing_error_final, min_val_loss, degree, l1_final, 0, i_final]
            SGDModel1 = pd.Series(SGDModel1, index = self.SGDModels.columns)
            self.SGDModels = self.SGDModels.append(SGDModel1, ignore_index=True)
            # SGD Ridge Regression
            print(f"Stochastic Gradient Descent for Ridge Regression || model degree {degree}")
            L2 = list(np.random.rand(5))
            min_val_loss = float("inf")
            training_error_final = float("inf")
            W_final = W
            l2_final = 0
            i_final = 0
            for l2 in L2:
                training_error, val_loss, W, i = self.SGD(X_train, X_val, Y_train, Y_val, l2=l2)
                if min_val_loss > val_loss:
                    min_val_loss = val_loss
                    W_final = W
                    training_error_final = training_error
                    l2_final = l2
                    i_final = i
            test_D = ((X_test @ W_final) - Y_test)
            testing_error_final = error = 0.5 * (np.sum(np.square(test_D)) + l2_final * np.sum(np.square(W_final)))
            SGDModel2 = [training_error_final, testing_error_final, min_val_loss, degree, 0, l2_final, i_final]
            SGDModel2 = pd.Series(SGDModel2, index = self.SGDModels.columns)
            self.SGDModels = self.SGDModels.append(SGDModel2, ignore_index=True)


# ## Building Polynomial Models

# In[3]:


models = Models(data)
models.buildModels(1, 10)


# ## Models

# In[4]:


models.GDModels


# In[5]:


models.SGDModels

