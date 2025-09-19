### MLR multiple linear regression
# y = X@beta + epsilon -> each y has a deterministic (X@beta -> E(Y|X)) and a random part (epsilon) -> y is a random variable
# y_hat = X@beta_hat -> model for E(Y|X), i.e., for the deterministic part of y
# the goal is to get a regression line that matches the population regression line
# X is of shape (n, p) with p > 1
# each beta_j is the effect of factor j **holding all other factors constant**

import numpy as np
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm

