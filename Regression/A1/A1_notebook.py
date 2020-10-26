#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[17]:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
plt.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# -------------------------------------------------------------------------------------------------------------------------------

# ## Load and Inspect Data

# In[18]:


data = pd.read_csv("insurance.txt")
data.head()


# In[19]:


data.info()


# In[20]:


data.describe()


# In[25]:


data.corr()


# -------------------------------------------------------------------------------------------------------------------------------

# ## Preprocess Data

# __The purpose of the following preprocessing is to demonstrate that the employed method is functioning correctly. However, while building regression models, the training and testing datasets are standardised separately to prevent data leakage.__

# In[5]:


column_names = list(data.columns)
standardize = lambda x : (x - x.mean()) / x.std() 


# __Standardize Data__

# In[6]:


for column in column_names:
    if column == "charges":
        continue
    data[column] = standardize(data[column])


# In[7]:


data.mean()


# Although standardised mean should be equal to 0, in reality they are infinitesimally close to zero. Double precision floats have 53 bits or 16 digits of precision, so this is expected behavior.

# In[8]:


data.std()


# In[9]:


x1, x2, x3, y = data["age"], data["bmi"], data["children"], data["charges"]


# -------------------------------------------------------------------------------------------------------------------------------

# ## Visualize Data

# In[10]:


plt.scatter(x1, y)
plt.xlabel("Age")
plt.ylabel("Charges");


# In[11]:


plt.scatter(x2, y)
plt.xlabel("BMI")
plt.ylabel("Charges");


# In[12]:


plt.scatter(x3, y)
plt.xlabel("Children")
plt.ylabel("Charges");


# -------------------------------------------------------------------------------------------------------------------------------

# ## Linear Regression Class

# In[16]:


class LinearRegression:
    def __init__(self, data):
        self.data = data
        self.models1 = pd.DataFrame(columns = ["w0", "w1", "w2", "w3", "testing_error", "training_error"])
        self.models2 = pd.DataFrame(columns = ["w0", "w1", "w2", "w3", "testing_error", "training_error", "iterations"])
        self.models3 = pd.DataFrame(columns = ["w0", "w1", "w2", "w3", "testing_error", "training_error", "iterations"])
        self.errors2 = []
        self.errors3 = []
    
    # shuffles data randomly
    def shuffle(self):
        self.data = self.data.sample(frac = 1)
    
    # splits a data frame in 70:30 ratio
    def split(self):
        train = self.data[:int(0.7 * len(data))]
        test = self.data[int(0.7 * len(data)):]
        return train, test
    
    # standardizes data
    def standardize(self, data):
        column_names = list(data.columns)
        standardize = lambda x : (x - x.mean()) / x.std()
        for column in column_names:
            if column == "charges":
                continue
            data[column] = standardize(data[column])
        return data
    
    # computes and returns X and Y matrices
    def matrices(self, train):
        x1, x2, x3, y = train["age"], train["bmi"], train["children"], train["charges"]
        x1, x2, x3= np.array(x1), np.array(x2), np.array(x3)
        Y = np.array(y)
        n = len(Y)
        Y = Y.reshape((n, 1))
        x0 = np.ones((n,1))
        x1 = np.reshape(x1, (n, 1))
        x2 = np.reshape(x2, (n, 1))
        x3 = np.reshape(x3, (n, 1))
        X = np.append(x0, x1, axis=1)
        X = np.append(X, x2, axis=1)
        X = np.append(X, x3, axis=1)
        return X, Y
    
    # Sum of Squares of Error Function
    def error(self, X, Y, w):
        Y_pred = np.dot(X, w)
        sose = (1/2) * np.sum(np.square(Y_pred-Y))
        return float(sose)
    
    # tests the model against testing data
    def testModel(self, test, w):
        Xt, Yt = self.matrices(test)
        testing_error = self.error(Xt, Yt, w)
        return testing_error
           
    # Performs Linear Regression by Normal Equations and returns corresponding Regression Weights
    def solve(self, X, Y):
        XT = np.transpose(X)
        A = XT.dot(X)
        A_inv = np.linalg.inv(A)
        b = XT.dot(Y)
        w = A_inv.dot(b)
        return w

    # Implements Gradient Descent Algorithm
    def gradientDescent(self, X, Y, w, learning_rate = 1e-7):
        i = 1
        errors = []
        error, prev_error = float("inf"), 0
        while abs(prev_error-error) > 1e-5:
            prev_error = error
            Y_pred = X.dot(w)
            gradient = np.dot(X.transpose(), Y_pred- Y)
            w -= learning_rate * gradient
            error = self.error(X, Y, w)
            errors.append(error)
            i+=1
        return w, errors, i

    # Implements Stochastic Gradient Descent Algorithm
    def stochasticGradientDescent(self, X, Y, w, learning_rate = 1e-5):
        m, i = len(Y), 0
        errors = []
        error, prev_error = float("inf"), 0
        while abs(prev_error-error) > 1e-2:
            prev_error = error
            x = X[i%m].reshape(1, 4)
            y = Y[i%m].reshape(1, 1)
            y_pred = x.dot(w)
            gradient = np.dot(x.transpose(), y_pred- y)
            w -= learning_rate * gradient
            error = self.error(X, Y, w)
            errors.append(error)
            i+=1
        return w, errors, i + 1

    # main function to build models
    def buildModels(self, number = 20):
        for i in range(number):
            self.shuffle()                                                                  # shuffle dataset
            train, test = self.split()                                                      # split dataset
            train = self.standardize(train)                                                 # standardise training data
            test = self.standardize(test)                                                   # standardise testing data
            X, Y = self.matrices(train)                                                     # obtain X and Y matrices
            w1 = self.solve(X, Y)                                                           # regression by normal equations
            w2 = np.random.rand(4, 1)                                                       # generate random weights w for grad. desc.
            w2, model_errors2 , iterations2 = self.gradientDescent(X, Y, w2)                # regression by gradient descent
            w3 = np.random.rand(4, 1)                                                       # generate random weights w for st. grad. desc.
            w3, model_errors3 , iterations3 = self.stochasticGradientDescent(X, Y, w3)      # regression by stochastic grad. desc.
            training_error1 = self.error(X, Y, w1)                                          # calculate training error
            training_error2 = self.error(X, Y, w2)                                          # calculate training error
            training_error3 = self.error(X, Y, w3)                                          # calculate training error
            testing_error1 = self.testModel(test, w1)                                       # calculate testing error
            testing_error2 = self.testModel(test, w2)                                       # calculate testing error
            testing_error3 = self.testModel(test, w3)                                       # calculate testing error
            # append the model, errors to models dataframe
            model1 = list(w1)
            model2 = list(w2)
            model3 = list(w3)
            model1.append(testing_error1)
            model2.append(testing_error2)
            model3.append(testing_error3)
            model1.append(training_error1)
            model2.append(training_error2)
            model3.append(training_error3)
            model2.append(iterations2)
            model3.append(iterations3)
            model1 = pd.Series(model1, index = self.models1.columns)
            model2 = pd.Series(model2, index = self.models2.columns)
            model3 = pd.Series(model3, index = self.models3.columns)
            self.models1 = self.models1.append(model1, ignore_index=True)
            self.models2 = self.models2.append(model2, ignore_index=True)
            self.models3 = self.models3.append(model3, ignore_index=True)
            self.errors2.append(model_errors2)
            self.errors3.append(model_errors3)


# -------------------------------------------------------------------------------------------------------------------------------

# ## Build Regression Models

# In[14]:


data = pd.read_csv("insurance.txt")


# In[15]:


lr_models = LinearRegression(data)
lr_models.buildModels(20)


# In[16]:


lr_models.models1


# In[17]:


lr_models.models2


# In[18]:


lr_models.models3


# <hr>

# ## Report

# ### Pre-Processing:
# __1. Shuffling the data:__ The dataset was randomly shuffled to ensure the split into train and test data remains random and the model is general.<br>
# __2. Splitting the data:__ The dataset was split into training and testing data in 7:3 ratio in order.<br>
# __3. Standardising the data:__ The feature attributes of the training and testing datasets were standardised separately to prevent leakage of testing data into training data. Finally, the Panda Series were converted into Numpy Arrays for easier and faster computation.<br>

# ### Describe Regression Models

# In[19]:


lr_models.models1.describe()


# In[20]:


lr_models.models2.describe()


# In[21]:


lr_models.models3.describe()


# ### Model:

# Three models were generated, one by solving Normal Equations, one by Gradient Descent and another by Stochastic Gradient Descent after each random shuffle and split of the dataset. The intercept of the models was close to 13000 as expected, which is the mean of the target attribute. Sum of squares of errors was used to represent the accuracy of the predictive model. All three algorithms yieled very similar predictive linear regression models.

# $y = w_{0} + w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3}$

# $x_{1}, x_{2}, x_{3}$ represent the age, bmi and number of children respectively. $ w_{1},w_{2}, w_{3}$ are weights associated with $x_{1}, x_{2}, x_{3}$.

# $$X = 
# \begin{bmatrix} 
# 1 & x_{11} & x_{12} & x_{13}\\
# 1 & x_{21} & x_{22} & x_{23}\\
# . & . & . & . \\
# . & . & . & . \\
# 1 & x_{m1} & x_{m2} & x_{m3}\\
# \end{bmatrix}
# \quad
# $$
# $$ $$
# $$Y = 
# \begin{bmatrix} 
# y_{1}\\
# y_{2}\\
# .\\
# .\\
# y_{m}\\
# \end{bmatrix}
# \quad
# $$ where $m$ = size of training data
# $$ $$
# $$ω = 
# \begin{bmatrix} 
# ω_{0}\\
# ω_{1}\\
# ω_{2}\\
# ω_{3}\\
# \end{bmatrix}
# \quad
# $$
# $$ $$
# $$b = X^{T} . Y$$

# $$Sum of Squares of Errors =
# E(ω) = 
# \begin{equation}
# \frac{1}{2} * \sum_{n=0}^{N} (x_{n}*ω - y_{n})^{2} 
# \end{equation}
# $$

# Matrix dot product, inversion were performed with Numpy library by first converting Pandas Series to Numpy Array.

# ### Algorithms:

# #### Linear Regression by Solving Normal Equations

# $$ ω = (X^{T}.X)^{-1}x^{T}.Y = (X^{T}.X)^{-1}.b$$

# #### Linear Regression by Gradient Descent

# $$ 
# \begin{equation}
# \frac{\partial E(ω)}{\partial ω} = (X.ω - Y).X
# \end{equation}
# $$
# $$ $$
# $$ Gradient$$
# $$ $$
# $$
# ω = ω - η * \frac{\partial E(ω)}{\partial ω}
# $$
# $$ $$
# where $η$ is the learning rate

# #### Linear Regression by Stochastic Gradient Descent

# SGD makes sequential passes over the training data, and during each pass, updates feature weights one example at a time with the aim of approaching the optimal weights that minimize the loss.

# $$ 
# \begin{equation}
# \frac{\partial E(ω)}{\partial ω}_{ω=ω_{n}} = (x_{n}.ω - y_{n}).x_{n}
# \end{equation}
# $$
# $$ $$
# $$ Gradient$$
# $$ $$
# $$
# ω = ω - η * \frac{\partial E(ω)}{\partial ω}_{ω=ω_{n}}
# $$
# $$ $$
# where $η$ is the learning rate

# ### Mean, Variance and Minimum of Training Error

# __Normal Equations__

# In[22]:


print(f"Mean of training error obtained over 20 regression models = {lr_models.models1.training_error.mean()}")


# In[23]:


print(f"Variance of training error obtained over 20 regression models = {lr_models.models1.training_error.var()}")


# In[24]:


print(f"Minimum training error obtained over 20 regression models = {lr_models.models1.training_error.min()}")


# __Gradient Descent__

# In[25]:


print(f"Mean of training error obtained over 20 regression models = {lr_models.models2.training_error.mean()}")


# In[26]:


print(f"Variance of training error obtained over 20 regression models = {lr_models.models2.training_error.var()}")


# In[27]:


print(f"Minimum training error obtained over 20 regression models = {lr_models.models2.training_error.min()}")


# __Stochastic Gradient Descent__

# In[28]:


print(f"Mean of training error obtained over 20 regression models = {lr_models.models3.training_error.mean()}")


# In[29]:


print(f"Variance of training error obtained over 20 regression models = {lr_models.models3.training_error.var()}")


# In[30]:


print(f"Minimum training error obtained over 20 regression models = {lr_models.models3.training_error.min()}")


# ### Mean, Variance and Mean of Testing Error

# __Normal Equations__

# In[31]:


print(f"Mean of testing error obtained over 20 regression models = {lr_models.models1.testing_error.mean()}")


# In[32]:


print(f"Variance of training error obtained over 20 regression models = {lr_models.models1.testing_error.var()}")


# In[33]:


print(f"Minimum testing error obtained over 20 regression models = {lr_models.models1.testing_error.min()}")


# __Gradient Descent__

# In[34]:


print(f"Mean of testing error obtained over 20 regression models = {lr_models.models2.testing_error.mean()}")


# In[35]:


print(f"Variance of training error obtained over 20 regression models = {lr_models.models2.testing_error.var()}")


# In[36]:


print(f"Minimum testing error obtained over 20 regression models = {lr_models.models2.testing_error.min()}")


# __Stochastic Gradient Descent__

# In[37]:


print(f"Mean of testing error obtained over 20 regression models = {lr_models.models3.testing_error.mean()}")


# In[38]:


print(f"Variance of training error obtained over 20 regression models = {lr_models.models3.testing_error.var()}")


# In[39]:


print(f"Minimum testing error obtained over 20 regression models = {lr_models.models3.testing_error.min()}")


# ### Plot the Convergence: Error vs Epochs

# __Plot of $E(\omega)$ against the number of iterations of Gradient Descent__

# In[40]:


errorsGD = np.array(lr_models.errors2, dtype=object)


# In[41]:


for error in errorsGD:
    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("$E(\omega)$")
    plt.title("Values of the Error Function over Iterations of Gradient Descent");


# __Plot of $E(\omega)$ against the number of iterations of Stochastic Gradient Descent__

# In[42]:


errorsSGD = np.array(lr_models.errors3, dtype=object)


# In[43]:


for error in errorsSGD:
    plt.plot(error)
    plt.xlabel("Iterations")
    plt.ylabel("$E(\omega)$")
    plt.title("Values of the Error Function over Iterations of Stochastic Gradient Descent");


# <hr>

# 1. All three methods yield very close models. Normal Equations and Gradient Descent methods differ only in decimals while Stochastic Gradient Descent differs by a small value. The result is due to the fact that the error function has a single global minima, and all three algorithms aim to converge to it. Since the minima is unique, the models converge to the same weights. However, the small discrepency is due to the fact that gradient and stochastic gradient descent algorithms are stopped prematurely when the difference in gradient calculated is too small to result in a significant change in the model. The most efficient algorithm to work with would be Stochastic Gradient Descent, which reaches the minima faster than other algorithms when the dataset is very large.

# 2. Standardization is a scaling technique where the values are centered around the mean i.e. mean of the dsitribution becomes zero with a unit standard deviation. In multivariate regression, standardizations brings variables having different units to comparable units, better ensuring that all the weights are updated at similar rates and a more accurate predictive model.

# 3. In general, increasing the number of training iterations tends to minimise the loss function and resulting in a more accurate model. However, after a large number of epochs, the change in error function becomes negligible and the computational complexity of running more iterations would outweigh the resulting predictive accuracy of the model.

# 4. Stochastic gradient descent plots convegred to the minima significantly faster than gradient descent algorithm. Approximately, it took SGD 8000 epochs, while it took GD 140,000 epochs to achieve the same. However, GD algorithm yielded more accurate model. In a real world data, SGD is more practical way of implementing linear regression.

# 5. If a very large learning rate is used in GD/SGD algorithm, the error value would overshoot the minima of the loss function. It may keep diverging from the minima indefinitely, thereby resulting in an infinite loop.

# 6. If the model does not have a bias term, it'll be equal to zero when all the variables are zero. However, the mean of the 'charges' variable is equal to 13000. Therefore such a model would not fit the data well and result in larger error values. The minima would have been larger without a bias term.

# 7. The final vector value signifies the linear model that best fits the training data, given initial hypothesis. Since the feature attributes are standardized, they update at similar rates. Noticeably, $ω_{1}$  has the largest value among all the weights exlcuding bias value, meaning the model is susceptible to higher change due to $ω_{1}$ given the same difference in all feature attributes. Therefore, 'age' has the highest influence on the target attribute. Similarly, it can be deduced that 'children' has the least infuence on the target attribute.
