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

class OLS:

    def __init__(self, pop_size, sample_size, sigma_epsilon, beta):
        self.pop_size = pop_size
        self.sample_size = sample_size
        self.sigma_epsilon = sigma_epsilon
        self.epsilon = np.random.normal(0, self.sigma_epsilon, (self.pop_size, ))
        self.beta = np.array(beta)

    def make_X(self):
        self.X = np.random.normal(20, 10, (self.pop_size, ))
        self.X = sm.add_constant(self.X)

    def make_y(self):
        self.y = self.X@self.beta + self.epsilon

    def make_X_y_sample(self):
        idx_sample = np.random.randint(low=0, high=len(self.X), size=self.sample_size)
        self.X_sample, self.y_sample = self.X[idx_sample], self.y[idx_sample]

    def make_models(self):
        self.model_pop = sm.OLS(self.y, self.X)
        self.model_pop = self.model_pop.fit()
        self.y_hat_pop = self.model_pop.predict(self.X)

        self.model_sample = sm.OLS(self.y_sample, self.X_sample)
        self.model_sample = self.model_sample.fit()
        self.y_hat_sample = self.model_sample.predict(self.X_sample)
        self.beta_hat = self.model_sample.params 

    def plot_population_vs_sample(self):

        fig, ax = plt.subplots()
        ax.scatter(self.X[:, 1], self.y, facecolors='gray', edgecolors='gray', s=10, linewidths=0.5, alpha=0.3)
        ax.scatter(self.X_sample[:, 1], self.y_sample, facecolor='None', edgecolors='red', linewidths=0.5, s=30)
        ax.plot(self.X[:, 1], self.y_hat_pop, color='gray')
        ax.plot(self.X_sample[:, 1], self.y_hat_sample, color='red')
        ax.text(x=-10, y=120, s=rf'$\beta_0 = {self.beta[0]:.3f}, \hat{{\beta}}_0 = {self.beta_hat[0]:.3f}$')
        ax.text(x=-10, y=108, s=rf'$\beta_1 = {self.beta[1]:.3f}, \hat{{\beta}}_1 = {self.beta_hat[1]:.3f}$')
        plt.savefig('ols_pop_vs_sample.png')

    def compute_residuals_stats(self):
        self.e_i = self.model_sample.resid
        self.e_i_mean = np.mean(self.e_i)
        self.e_i_sd = np.sqrt(np.var(self.e_i, ddof=2)) # this is rse
        self.e_i_x = np.linspace(self.e_i.min(), self.e_i.max(), 300)
        self.e_i_pdf = scipy.stats.norm.pdf(self.e_i_x, loc=self.e_i_mean, scale=self.e_i_sd)
        self.y_hat_i = self.model_sample.fittedvalues

    def print_rss_relative_to_y_mean(self):
        print(f'RSS relative to y_meam: {self.e_i_sd:.2f} --- {self.y_sample.mean():.2f}')
        print(f'Real Var(epsilon) = {self.sigma_epsilon:.2f} --- RSE = {self.e_i_sd:.2f}')

    def compute_diagnostics(self):
        self.omnibus_stat, self.omnibus_p = omni_normtest(self.e_i) 
        self.jb_stat, self.jb_p, self.skew, self.kurtosis = jarque_bera(self.e_i)
        lb = acorr_ljungbox(self.e_i, lags=[1], return_df=True)
        self.lb_stat, self.lb_pvalue = lb['lb_stat'].values[0], lb['lb_pvalue'].values[0]
        influence = OLSInfluence(self.model_sample)
        self.leverage = influence.hat_matrix_diag
        self.studentized_resid = influence.resid_studentized_external

    def plot_diagnostics(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].set_title(r'Distr. $e_i$')
        ax[0].hist(self.e_i, density=True)
        ax[0].plot(self.e_i_x, self.e_i_pdf)
        ax[0].text(
            0.025, 0.95, 
            f'Omnibus: {self.omnibus_stat:.2f}({self.omnibus_p:.2f})\nJarque-Bera: {self.jb_stat:.2f}({self.jb_p:.2f})\nSkew.: {self.skew:.2f}\nKurt.:{self.kurtosis:.2f}', 
            transform=ax[0].transAxes, verticalalignment='top'
        )
        ax[0].set_xlabel(r'$e_i$')
        ax[0].set_ylabel('Density')
        ax[1].set_title(r'Fitted vals. vs. $e_i$')
        ax[1].scatter(self.y_hat_i, self.e_i)
        ax[1].axhline(y=0, color='black', linestyle='--')
        ax[1].set_xlabel('Fitted vals.')
        ax[1].set_ylabel(r'$e_i$')
        ax[1].text(0.025, 0.95, f'Ljung-Box:{self.lb_stat:.2f}({self.lb_pvalue:.2f})', transform=ax[1].transAxes, verticalalignment='top')
        ax[2].set_title(r'Leverage vs. studentized $e_i$')
        ax[2].scatter(self.leverage, self.studentized_resid)
        ax[2].set_xlabel('Leverage')
        ax[2].set_ylabel(r'Studentized $e_i$')
        plt.tight_layout()
        plt.savefig('ols_distr_ei.png')


    def run(self):
        self.make_X()
        self.make_y()
        self.make_X_y_sample()
        self.make_models()
        self.plot_population_vs_sample()
        self.compute_residuals_stats()
        self.print_rss_relative_to_y_mean()
        self.compute_diagnostics()
        self.plot_diagnostics()


if __name__ == '__main__':
    ols = OLS(1000, 200, 10, [1.44, -2.43])
    ols.run()




