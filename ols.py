import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

### OLS 
# y = beta_0 + beta_1*x + epsilon -> population regression line
#
pop_size = 1000
x = np.random.normal(20, 10, pop_size).reshape((pop_size, 1))
sigma_epsilon = 5
epsilon = np.random.normal(0, sigma_epsilon, pop_size)
beta_0 = 2.3324
beta_1 = 1.0234
y = beta_0 + beta_1*x[:, 0] + epsilon

n = 20
idx_sample = np.random.randint(low=0, high=len(x), size=n)
x_sample, y_sample = x[idx_sample], y[idx_sample]


model_pop = LinearRegression()
model_pop.fit(x, y)
y_mean_pop = model_pop.predict(x)

model = LinearRegression()
model.fit(x_sample, y_sample)
y_mean_sample = model.predict(x_sample)

beta_0_hat = model.intercept_
beta_1_hat = model.coef_[0]


fig, ax = plt.subplots()
ax.scatter(x[:, 0], y, facecolors='gray', edgecolors='gray', s=10, linewidths=0.5, alpha=0.3)
ax.scatter(x_sample[:, 0], y_sample, facecolor='None', edgecolors='red', linewidths=0.5, s=50)
ax.plot(x[:, 0], y_mean_pop, color='gray')
ax.plot(x_sample[:, 0], y_mean_sample, color='red')
ax.text(x=-5, y=50, s=rf'$\beta_0 = {beta_0:.3f}, \hat{{\beta}}_0 = {beta_0_hat:.3f}$')
ax.text(x=-5, y=40, s=rf'$\beta_1 = {beta_1:.3f}, \hat{{\beta}}_1 = {beta_1_hat:.3f}$')
plt.savefig('ols.png')
