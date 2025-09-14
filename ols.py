import numpy as np
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import omni_normtest, jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence

### OLS 
# y = X@beta + epsilon -> each y has a deterministic (X@beta -> E(Y|X)) and a random part (epsilon) -> y is a random variable
# y_hat = X@beta_hat -> model for E(Y|X), i.e., for the deterministic part of y
# the goal is to get a regression line that matches the population regression line

pop_size = 5000
X = np.random.normal(20, 10, (pop_size, ))
X = sm.add_constant(X)
sigma_epsilon = 7
epsilon = np.random.normal(0, sigma_epsilon, (pop_size, ))
beta = np.array([1.0234, 2.3324]) 
y = X@beta + epsilon

n = 60
idx_sample = np.random.randint(low=0, high=len(X), size=n)
X_sample, y_sample = X[idx_sample], y[idx_sample]

model_pop = sm.OLS(y, X)
model_pop = model_pop.fit()
y_hat_pop = model_pop.predict(X)

model_sample = sm.OLS(y_sample, X_sample)
model_sample = model_sample.fit()
y_hat_sample = model_sample.predict(X_sample)
beta_hat = model_sample.params 

# population vs. sample OLS
fig, ax = plt.subplots()
ax.scatter(X[:, 1], y, facecolors='gray', edgecolors='gray', s=10, linewidths=0.5, alpha=0.3)
ax.scatter(X_sample[:, 1], y_sample, facecolor='None', edgecolors='red', linewidths=0.5, s=30)
ax.plot(X[:, 1], y_hat_pop, color='gray')
ax.plot(X_sample[:, 1], y_hat_sample, color='red')
ax.text(x=-10, y=120, s=rf'$\beta_0 = {beta[0]:.3f}, \hat{{\beta}}_0 = {beta_hat[0]:.3f}$')
ax.text(x=-10, y=108, s=rf'$\beta_1 = {beta[1]:.3f}, \hat{{\beta}}_1 = {beta_hat[1]:.3f}$')
plt.savefig('ols_pop_vs_sample.png')

# residuals and fitted values
e_i = model_sample.resid
e_i_mean = np.mean(e_i)
e_i_sd = np.std(e_i)
e_i_x = np.linspace(e_i.min(), e_i.max(), 300)
e_i_pdf = scipy.stats.norm.pdf(e_i_x, loc=e_i_mean, scale=e_i_sd)
y_hat_i = model_sample.fittedvalues

print(f'RSS relative to y_meam: {e_i_sd:.2f} --- {y_sample.mean():.2f}')
print(f'Real Var(epsilon) = {sigma_epsilon:.2f} --- RSE = {e_i_sd:.2f}')

# regression diagnistics
omnibus_stat, omnibus_p = omni_normtest(e_i) 
jb_stat, jb_p, skew, kurtosis = jarque_bera(e_i)
lb = acorr_ljungbox(e_i, lags=[1], return_df=True)
lb_stat, lb_pvalue = lb['lb_stat'].values[0], lb['lb_pvalue'].values[0]
influence = OLSInfluence(model_sample)
leverage = influence.hat_matrix_diag
studentized_resid = influence.resid_studentized_external

# plot diagnostics
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].set_title(r'Distr. $e_i$')
ax[0].hist(e_i, density=True)
ax[0].plot(e_i_x, e_i_pdf)
ax[0].text(
    0.025, 0.95, 
    f'Omnibus: {omnibus_stat:.2f}({omnibus_p:.2f})\nJarque-Bera: {jb_stat:.2f}({jb_p:.2f})\nSkew.: {skew:.2f}\nKurt.:{kurtosis:.2f}', 
    transform=ax[0].transAxes, verticalalignment='top'
)
ax[0].set_xlabel(r'$e_i$')
ax[0].set_ylabel('Density')
ax[1].set_title(r'Fitted vals. vs. $e_i$')
ax[1].scatter(y_hat_i, e_i)
ax[1].axhline(y=0, color='black', linestyle='--')
ax[1].set_xlabel('Fitted vals.')
ax[1].set_ylabel(r'$e_i$')
ax[1].text(0.025, 0.95, f'Ljung-Box:{lb_stat:.2f}({lb_pvalue:.2f})', transform=ax[1].transAxes, verticalalignment='top')
ax[2].set_title(r'Leverage vs. studentized $e_i$')
ax[2].scatter(leverage, studentized_resid)
ax[2].set_xlabel('Leverage')
ax[2].set_ylabel(r'Studentized $e_i$')
plt.tight_layout()
plt.savefig('ols_distr_ei.png')


