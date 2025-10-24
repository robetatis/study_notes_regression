### MLR multiple linear regression
# y = X@beta + epsilon -> each y has a deterministic (X@beta -> E(Y|X)) and a random part (epsilon) -> y is a random variable
# y_hat = X@beta_hat -> model for E(Y|X), i.e., for the deterministic part of y
# the goal is to get a regression line that matches the population regression line
# X is of shape (n, p) with p > 1
# each beta_j is the effect of factor j **holding all other factors constant**

import pandas as pd
import numpy as np
import scipy
from pygam import LinearGAM, s
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, root_mean_squared_error
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import omni_normtest, jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor


class MLR:

    def model_basic(self, pop_size, sample_size, sigma_epsilon, beta):
        self.pop_size = pop_size
        self.sample_size = sample_size
        self.sigma_epsilon = sigma_epsilon
        self.epsilon = np.random.normal(0, self.sigma_epsilon, (self.pop_size, ))
        self.beta = np.array(beta)        
        self.p = len(self.beta) - 1

        self.X = np.random.normal(20, 10, (self.pop_size, self.p))
        self.X = sm.add_constant(self.X)
        self.y = self.X@self.beta + self.epsilon

        idx_sample = np.random.randint(low=0, high=len(self.X), size=self.sample_size)
        self.X_sample, self.y_sample = self.X[idx_sample], self.y[idx_sample]

        self.model_pop = sm.OLS(self.y, self.X)
        self.model_pop = self.model_pop.fit()
        self.y_hat_pop = self.model_pop.predict(self.X)

        self.model_sample = sm.OLS(self.y_sample, self.X_sample)
        self.model_sample = self.model_sample.fit()
        self.y_hat_sample = self.model_sample.predict(self.X_sample)
        self.beta_hat = self.model_sample.params 

        self.plot_population_vs_sample()
        self.compute_residuals_stats()
        self.print_rss_relative_to_y_mean()
        self.compute_diagnostics()
        self.plot_diagnostics(filename='mlr_distr_ei.png')

    def model_interaction(self, filename):
        n = 500
        beta = [1.3, 2.4, 4.2, 4.1]
        X1 = np.random.normal(10, 20, n)
        X2 = np.random.normal(10, 20, n)
        X1X2 = X1*X2
        y = beta[0] + beta[1]*X1 + beta[2]*X2 + beta[3]*X1X2 + np.random.normal(0, 10, n)
        X_with_interaction = np.column_stack([X1, X2, X1X2])
        X_with_interaction = sm.add_constant(X_with_interaction)
        X_no_interaction = np.column_stack([X1, X2])
        X_no_interaction = sm.add_constant(X_no_interaction)        

        self.model_sample = sm.OLS(y, X_with_interaction)
        self.model_sample = self.model_sample.fit()
        self.y_hat = self.model_sample.predict(X_with_interaction)

        print(self.model_sample.summary())
        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_diagnostics(filename)

    def model_categorical_predictors(self, filename_scatterplot, filename_diagnostics):
        n = 500
        sigma_epsilon = 80
        epsilon = np.random.normal(0, sigma_epsilon, n)
        beta = np.array([1.4, 3.4, 230.4])
        X1 = np.random.normal(20, 50, n)
        X2 = np.random.choice([0, 1], n)
        X = sm.add_constant(np.column_stack([X1, X2]))
        y = X@beta + epsilon

        self.model_sample = sm.OLS(y, X)
        self.model_sample = self.model_sample.fit()
        self.y_hat = self.model_sample.predict(X)

        print(self.model_sample.summary())
        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_diagnostics(filename_diagnostics)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 1], y, edgecolors=np.where(X[:, 2]==1, 'black', 'red'), s=20, facecolor='None')
        for i in np.unique(X[:, 2]):
            ax.plot(X[:, 1][X[:, 2] == i], self.y_hat[X[:, 2] == i], color='black' if i==1 else 'red')
        plt.savefig(filename_scatterplot)

    def model_interaction_categorical_predictors(self, filename_scatterplot, filename_diagnostics):
        n = 500
        sigma_epsilon = 100
        beta = np.array([3.4, 2.1, 300.1, 5.4])
        epsilon = np.random.normal(0, sigma_epsilon, n)
        X1 = np.random.normal(50, 50, n)
        X2 = np.random.choice([0, 1], n)
        X3 = X1*X2
        X = sm.add_constant(np.column_stack([X1, X2, X3]))
        y = X@beta + epsilon

        self.model_sample = sm.OLS(y, X)
        self.model_sample = self.model_sample.fit()
        self.y_hat = self.model_sample.predict(X)
        print(self.model_sample.summary())

        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_diagnostics(filename_diagnostics)

        fig, ax = plt.subplots()
        ax.scatter(X[:, 1], y, edgecolors=np.where(X[:, 2]==1, 'black', 'red'), s=20, facecolor='None')
        for i in np.unique(X[:, 2]):
            ax.plot(X[:, 1][X[:, 2] == i], self.y_hat[X[:, 2] == i], color='black' if i==1 else 'red')
        plt.savefig(filename_scatterplot)

    def model_missing_nonlinear_term(self, filename_diagnostics_ok, filename_diagnostics_missing):
        n = 500
        sigma_epsilon = 300
        epsilon = np.random.normal(0, sigma_epsilon, n)
        beta = [1.3, 0.4, 1.33]
        X1 = np.random.normal(2, 10, n)
        X2 = X1**2
        X_real = sm.add_constant(np.column_stack([X1, X2]))
        y = X_real@beta + epsilon

        X_missing_nonlinear = X_real[:, :-1]
        self.model_sample = sm.OLS(y, X_missing_nonlinear)
        self.model_sample = self.model_sample.fit()
        self.y_hat = self.model_sample.predict(X_missing_nonlinear)
        print(self.model_sample.summary())

        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_diagnostics(filename_diagnostics_missing)

        self.model_sample = sm.OLS(y, X_real)
        self.model_sample = self.model_sample.fit()
        self.y_hat = self.model_sample.predict(X_real)
        print(self.model_sample.summary())

        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_diagnostics(filename_diagnostics_ok)

    def model_heteroscedastic(self, filename_diagnostics_heteroscedastic):       
        n = 500
        sigma_epsilon = 3
        X = np.random.uniform(10, 100, n)
        epsilon_heteroscedastic = np.random.normal(0, sigma_epsilon*X**(0.66))
        beta = [1.3, 2.4]
        X = sm.add_constant(X)
        y = X@beta + epsilon_heteroscedastic

        self.model_sample = sm.OLS(y, X)
        self.model_sample = self.model_sample.fit()
        self.y_hat = self.model_sample.predict(X)
        print(self.model_sample.summary())

        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_diagnostics(filename_diagnostics_heteroscedastic)

    def plot_population_vs_sample(self):
        fig, ax = plt.subplots()
        ax.scatter(self.y_hat_pop, self.y, facecolors='gray', edgecolors='gray', s=10, linewidths=0.5, alpha=0.3)
        ax.scatter(self.y_hat_sample, self.y_sample, facecolor='None', edgecolors='red', linewidths=0.5, s=30)
        ax.plot(self.y, self.y, color='gray')
        ax.plot(self.y_sample, self.y_sample, color='red')
        for i in range(len(self.beta)):
            ax.text(x=0.02, y=0.93 - 0.075*i, transform=ax.transAxes, s=rf'$\beta_{i} = {self.beta[i]:.3f}, \hat{{\beta}}_{i} = {self.beta_hat[i]:.3f}$')        
        ax.set_ylabel('y pred')
        ax.set_xlabel('y obs')
        plt.savefig('mlr_pop_vs_sample.png')

    def compute_residuals_stats(self):
        self.e_i_mean = np.mean(self.e_i)
        self.e_i_sd = np.sqrt(np.var(self.e_i, ddof=2)) # this is rse (the residuals' standard error)
        self.e_i_x = np.linspace(self.e_i.min(), self.e_i.max(), 300)
        self.e_i_pdf = scipy.stats.norm.pdf(self.e_i_x, loc=self.e_i_mean, scale=self.e_i_sd)

    def print_rss_relative_to_y_mean(self):
        print(f'RSS relative to y_meam: {self.e_i_sd:.2f} --- {self.y_sample.mean():.2f}')
        print(f'Real Var(epsilon) = {self.sigma_epsilon:.2f} --- RSE = {self.e_i_sd:.2f}')

    def compute_diagnostics(self):

        self.omnibus_stat, self.omnibus_p = omni_normtest(self.e_i) 
        self.jb_stat, self.jb_p, self.skew, self.kurtosis = jarque_bera(self.e_i)
        lb = acorr_ljungbox(self.e_i, lags=[1], return_df=True)
        self.lb_stat, self.lb_pvalue = lb['lb_stat'].values[0], lb['lb_pvalue'].values[0]

        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.leverage = np.einsum('ij,jk,ik->i', self.X, XtX_inv, self.X)
        mse = np.mean(self.e_i**2)
        self.studentized_resid = self.e_i / np.sqrt(mse * (1 - self.leverage))

    def plot_diagnostics(self, filename):
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
        ax[1].scatter(self.y_hat, self.e_i, alpha=0.3)
        ax[1].axhline(y=0, color='black', linestyle='--')
        ax[1].set_xlabel('Fitted vals.')
        ax[1].set_ylabel(r'Studentized $e_i$')
        ax[1].text(0.025, 0.95, f'Ljung-Box:{self.lb_stat:.2f}({self.lb_pvalue:.2f})', transform=ax[1].transAxes, verticalalignment='top')

        ax[2].set_title(r'Leverage vs. studentized $e_i$')
        ax[2].scatter(self.leverage, self.studentized_resid, alpha=0.3)
        ax[2].set_xlabel('Leverage')
        ax[2].set_ylabel(r'Studentized $e_i$')

        plt.tight_layout()
        plt.savefig(filename)

    def plot_y_obs_vs_y_pred(self, filename):

        ymin = np.min([self.y, self.y_hat])
        ymax = np.max([self.y, self.y_hat])

        fig, ax = plt.subplots()
        ax.scatter(self.y, self.y_hat, alpha=0.3)
        ax.plot([ymin, ymax], [ymin, ymax], 'k--')
        ax.set_ylabel('y pred')
        ax.set_xlabel('y obs')
        plt.savefig(filename)

    def variance_inflation_colinear_X(self, X_is_colinear):
        
        X = sm.add_constant(np.array([[1, 0], [0, 1], [0, 0]]))
        
        if X_is_colinear:
            X = sm.add_constant(np.array([[1, 2.05], [2, 3.95], [3, 6.1]]))
        
        beta = np.array([1.5, 2.3, 0.8])
        sigma_epsilon = 0.05
        epsilon = np.random.normal(0, sigma_epsilon, 3)
        y = X@beta + epsilon

        beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
        var_beta_hat = sigma_epsilon**2 * np.linalg.inv(X.T @ X)

        r_reg = np.corrcoef(X[:, 1], X[:, 2])[0, 1]
        print(f'corr. btw. regressors = {r_reg:.4f}') 
        print(f'beta_hat = {beta_hat[0]:.4f}, {beta_hat[1]:.4f}, {beta_hat[2]:.4f}')
        print(f'var(beta_hat) = {var_beta_hat[0,0]:.4f}, {var_beta_hat[1,1]:.4f}, {var_beta_hat[2,2]:.4f}')

    def diagnose_collinearity(self, center=False):

        self.compute_vif()

        X = self.X

        # center X
        if center:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # compute X^TX
        XTX = X.T @ X

        # eigen-decomposition of X^TX
        eigvals, eigvects = np.linalg.eigh(XTX)

        eigvects = pd.DataFrame(eigvects, columns=eigvals, index=self.X.columns)
        
        # compute condition number lambda_max/lambda_min
        kappa = np.max(eigvals)/np.min(eigvals)
        print(f'Eigenvalues:\n{eigvals}')
        print(f'Condition number (kappa) = {kappa}')
        print(f'Eigenvectors: \n{eigvects}')

    def collinearity_3d_example(self):
        np.random.seed(0)
        X = np.random.randn(300, 2) @ np.array([[1,   0, 0.8],
                                                [0,   1,   0],
                                                [0.8, 0,   1]])

        XTX = X.T @ X

        eigvals, eigvecs = np.linalg.eigh(XTX)

        print("Eigenvalues:", eigvals)
        print("Eigenvectors (columns):\n", eigvecs)

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.3, label='Rows of X (observations)')

        origin = np.zeros(3)
        colors = ['r', 'g', 'b']
        for i in range(3):
            vec = XTX[:, i]
            ax.quiver(*origin, *vec, color=colors[i], arrow_length_ratio=0.1, label=f'col {i+1} of X^T X')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.legend()
        plt.tight_layout()
        plt.savefig('mlr_3d_example_collinearity_dependent.png')

    def collinearity_2d_example(self):

        base_noise = np.random.randn(100, 2)
        var_covar = np.array([[1, 0], [0, 1]])
        X = base_noise @ var_covar
        XTX = X.T @ X
        evals, evects = np.linalg.eigh(XTX)

        print(f'X^T X:\n{XTX}')
        print(f'\nEigenvalues:\n{evals}')
        print(f'\nEigenvectors:\n{evects}')

        origin = np.zeros(2)
        scale = 1
        xmin, xmax, ymin, ymax = 1.05*np.min(X[:, 0]), 1.05*np.max(X[:, 0]), 1.05*np.min(X[:, 1]), 1.05*np.max(X[:, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(X[:, 0], X[:, 1], edgecolors='black', facecolor='none', label=r'$X$')
        for i in range(XTX.shape[1]):
            ax.plot([0, XTX[0, i]], [0, XTX[1, i]], color='blue' if i == 0 else 'green', linestyle='--', label=rf'$X^TX_{{\cdot {i+1}}}$')
        ax.quiver(
            np.zeros(evects.shape[1]),
            np.zeros(evects.shape[1]),
            evects[0, :]*scale,
            evects[1, :]*scale,
            angles='xy', scale_units='xy', scale=1,
            color='r',
            width=0.005
        )
        ax.axvline(x=0, linestyle=':', color='gray')
        ax.axhline(y=0, linestyle=':', color='gray')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')
        arrow_handle = mlines.Line2D([], [], color='r', marker=r'$\rightarrow$', label='Eigenvectors')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(arrow_handle)
        labels.append('Eigenvectors')
        ax.legend(handles, labels)
        plt.savefig('mlr_2d_example_collinearity_independent.png')

    def y_distribution(self, bins, filename):
        
        fig, ax = plt.subplots()
        ax.hist(self.y, bins=bins, density=True)
        ax.set_title('Histogram of y')
        ax.set_xlabel('y')
        ax.set_ylabel('Density')
        plt.savefig(filename)

    def fit_mlr(self):
        columns = [
            'MedInc',
            'HouseAge',
            'Latitude',
            'Longitude',
            '_log1pTransformed_AveRooms',
            '_log1pTransformed_AveBedrms',
            '_log1pTransformed_AveOccup',
            '_log1pTransformed_Population'
        ]

        self.df = pd.concat([self.y, self.X[columns]], axis=1)
        formula = f"med_house_val_100k ~ {' + '.join(self.df.columns[1:])}"
        self.model_sample = smf.ols(formula, data=self.df).fit()
        self.y_hat = self.model_sample.fittedvalues
        self.e_i = self.y.values.ravel() - self.y_hat
        print(self.model_sample.summary())

    def plot_groups(self):
        group = self.X['geo_bin'].unique()[0]
        X_group = self.X[self.X['geo_bin'] == group]
        y_group = self.y[self.X['geo_bin'] == group]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.scatter(X_group['MedInc'], y_group, alpha=0.3, s=3)
        plt.savefig('mlr_groups_california_housing.png')

    def fit_mixed_effects_model(self):
        regressors = ['geo_bin', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup'] # 
        self.df = pd.concat([self.y, self.X[regressors]], axis=1)
        formula = "med_house_val_100k ~ MedInc + HouseAge + AveRooms + AveBedrms + Population + AveOccup"
        self.model_sample = smf.mixedlm(formula, self.df, groups=self.df["geo_bin"]).fit()
        self.y_hat = self.model_sample.fittedvalues
        self.e_i = self.y.values.ravel() - self.y_hat
        print(self.model_sample.summary())

    def compute_vif(self):
        self.vif = pd.DataFrame()
        self.vif['feature'] = self.X.columns
        self.vif['vif'] = [variance_inflation_factor(self.X.values, i)  for i in range(self.X.shape[1])]
        print(self.vif)

    def fit_ridge_regression(self):

        scaler = StandardScaler()
        self.X_centered = scaler.fit_transform(self.X)

        ridge = Ridge(alpha=1.0) # regularization strength
        ridge.fit(self.X_centered, self.y)

        self.y_hat = ridge.predict(self.X_centered)
        self.e_i = self.y - self.y_hat

        # print beta and se(beta)
        n, p = self.X_centered.shape

        sigma2 = np.sum(self.e_i**2) / (n - p)

        XtX = self.X_centered.T @ self.X_centered

        ridge_cov = sigma2 * np.linalg.inv(XtX + ridge.alpha * np.eye(p)) @ XtX @ np.linalg.inv(XtX + ridge.alpha * np.eye(p))

        ridge_se = np.sqrt(np.diag(ridge_cov))

        coef_summary = pd.DataFrame({
            "Coefficient": ridge.coef_.ravel(),
            "Std. Error": ridge_se
        })
        print(coef_summary)

    def rj_vs_ei(self, filename):

        # rj (residuals of Xj vs. all other regressors X_j) vs. e_i. meant to show which Xj drives residuals of full model
        regressors = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        model = LinearRegression()
        model.fit(self.X[regressors], self.y)
        y_hat = model.predict(self.X[regressors])
        e_i = self.y.values.ravel() - y_hat

        r_j = list()
        for j in regressors:
            Xj = self.X[j]
            X_except_j = self.X.drop(j, axis=1, inplace=False)
            model = LinearRegression()
            model.fit(X_except_j, Xj)
            Xj_hat = model.predict(X_except_j)
            r_j.append(Xj - Xj_hat)

        counter = 0
        fig, ax = plt.subplots(2, 4, figsize=(7, 4))
        for i in range(2):
            for j in range(4):
                Xj = self.X.iloc[:, counter]
                ax[i, j].scatter(r_j[counter], e_i, s=1, alpha=0.3)
                ax[i, j].set_title(self.X.columns[counter])
                counter += 1
        plt.tight_layout()
        plt.savefig(filename)

    def partial_residuals_plots(self, filename):

        # Xj vs. r_yj = ei + Xj*beta_j, where r_yj = ei + Xj*beta_j = residuals of full model (ei) + variation uniquely explained by Xj
        regressors = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
        model = LinearRegression()
        model.fit(self.X[regressors], self.y)
        y_hat = model.predict(self.X[regressors])
        e_i = self.y.values.ravel() - y_hat
        
        r_y = list()
        for j in range(len(model.coef_)):
            r_y_j = e_i + self.X.iloc[:, j] * model.coef_[j]
            r_y.append(r_y_j)

        counter = 0
        fig, ax = plt.subplots(2, 4, figsize=(7, 4))
        for i in range(2):
            for j in range(4):
                Xj = self.X.iloc[:, counter]
                ax[i, j].scatter(Xj, r_y[counter], s=1, alpha=0.3)
                ax[i, j].set_title(self.X.columns[counter])
                counter += 1
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def fit_mlr_using_principal_components(self):
        scaler = StandardScaler()
        X_centered = scaler.fit_transform(self.X)
        pca = PCA()
        self.X = pd.DataFrame(pca.fit_transform(X_centered), columns=[f'PC{i}' for i, _ in enumerate(self.X.columns)])
        self.df = pd.concat([self.y, self.X], axis=1)
        formula = f"med_house_val_100k ~ {' + '.join(self.df.columns[1:])}"
        self.model_sample = smf.ols(formula, data=self.df).fit()
        self.y_hat = self.model_sample.fittedvalues
        self.e_i = self.y.values.ravel() - self.y_hat
        print(self.model_sample.summary())

    def fit_linear_gam(self):

        X_coords = self.X[['Latitude', 'Longitude']].values
        y = self.y.values.ravel()
        gam = LinearGAM(s(0, n_splines=10) + s(1, n_splines=10)).fit(X_coords, y)
        self.y_hat = gam.predict(X_coords)
        self.e_i = self.y.values.ravel() - self.y_hat

    def fit_partial_least_squares(self):
        columns = [
            'MedInc',
            'HouseAge',
            'Latitude',
            'Longitude',
            '_log1pTransformed_AveRooms',
            '_log1pTransformed_AveBedrms',
            '_log1pTransformed_AveOccup',
            '_log1pTransformed_Population'
        ]

        scaler = StandardScaler()
        X_centered = scaler.fit_transform(self.X[columns])
        pls = PLSRegression(n_components=8)
        pls.fit(X_centered, self.y)
        self.y_hat = pls.predict(X_centered).ravel()
        self.e_i = self.y.values.ravel() - self.y_hat

        print(f"R² = {r2_score(self.y, self.y_hat):.3f}")
        print(f"RMSE = {root_mean_squared_error(self.y, self.y_hat):.3f}")

        fig, ax = plt.subplots()
        ax.scatter(self.y_hat, self.e_i, alpha=0.3)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_xlabel("Fitted values (y_hat)")
        ax.set_ylabel("Residuals")
        ax.set_title("PLS: Fitted vs. Residuals")
        plt.savefig('mlr_partial_least_squares_e_i.png')

    def make_geo_bins(self):
        lat_bin = pd.cut(self.X['Latitude'], bins=10)
        lon_bin = pd.cut(self.X['Longitude'], bins=10)
        self.X['geo_bin'] = lat_bin.astype(str) + '_' + lon_bin.astype(str)

    def preprocess(self):

        self.X = pd.DataFrame(self.ch.data, columns = self.ch.feature_names)
        self.y = pd.DataFrame(self.ch.target, columns=['med_house_val_100k'])

        # remove high-leverage points
        idx = [1979, 13366]
        self.y = self.y.drop(index=idx)
        self.X = self.X.drop(index=idx)

        # mask some problematic datapoints
        mask = (
            (self.y['med_house_val_100k'] < 5) & # remove censored data y>=5
            (self.X['HouseAge'] < 52) & # remove censored data HouseAge > 50
            (self.X['AveOccup'] < 50) # remove obvious data error AveOccup > 1000
        )
        self.X = self.X.loc[mask]
        self.y = self.y.loc[mask]

        # scale population
        self.X['Population'] = self.X['Population']/1000

        # turn y into pd series
        self.y = self.y['med_house_val_100k']

        # reset indices
        self.X = self.X.reset_index(drop=True)
        self.y = self.y.reset_index(drop=True)

        # log-transform right-skewed variables
        self.X['_log1pTransformed_AveRooms'] = self.X['AveRooms'].apply(lambda x: np.log1p(x))
        self.X['_log1pTransformed_AveBedrms'] = self.X['AveBedrms'].apply(lambda x: np.log1p(x))
        self.X['_log1pTransformed_AveOccup'] = self.X['AveOccup'].apply(lambda x: np.log1p(x))
        self.X['_log1pTransformed_Population'] = self.X['Population'].apply(lambda x: np.log1p(x))
        
    def plot_histograms_X(self):
        columns = [
            'MedInc',
            'HouseAge',
            'Latitude',
            'Longitude',
            '_log1pTransformed_AveRooms',
            '_log1pTransformed_AveBedrms',
            '_log1pTransformed_AveOccup',
            '_log1pTransformed_Population'
        ]
        counter = 0
        fig, ax = plt.subplots(2, 4, figsize=(7, 4))
        for i in range(2):
            for j in range(4):
                Xj = self.X[columns[counter]]
                ax[i, j].hist(Xj, density=True)
                ax[i, j].set_title(columns[counter])
                counter += 1
        plt.tight_layout()
        plt.savefig('mlr_hist_X_california_housing.png')

    def fit_mixed_effects_model_pca(self):

        columns = [
            'MedInc',
            'HouseAge',
            'Latitude',
            'Longitude',
            '_log1pTransformed_AveRooms',
            '_log1pTransformed_AveBedrms',
            '_log1pTransformed_AveOccup',
            '_log1pTransformed_Population'
        ]

        # create orthogonal version of X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X[columns])          # self.X has Latitude & Longitude columns
        pca = PCA(n_components=min(8, X_scaled.shape[1]))
        X_pca = pca.fit_transform(X_scaled)
        X_pca = pd.DataFrame(X_pca, columns=[f'PC{i}' for i in range(X_pca.shape[1])])
        X_pca = X_pca.set_index(self.X.index)

        # create geo_bins
        bins_lat = 10
        bins_lon = 10
        lat = self.X['Latitude'].values
        lon = self.X['Longitude'].values
        lat_bin = np.digitize(lat, np.linspace(lat.min(), lat.max(), bins_lat))
        lon_bin = np.digitize(lon, np.linspace(lon.min(), lon.max(), bins_lon))
        groups = pd.Series((lat_bin - 1) * bins_lon + (lon_bin - 1), name='bin')
        groups.index = self.X.index
        
        # convert X_pca into deviations from geo-bin means        
        group_means = X_pca.groupby(groups).transform('mean')
        X_within = X_pca - group_means
        X_within = X_within.reset_index(drop=True)

        # prepare data frame for mixed model
        self.df = pd.concat([
            pd.Series(self.y.values.ravel(), name='y').reset_index(drop=True),
            X_within,
            groups.reset_index(drop=True)
        ], axis=1)

        # fit mixed model: random intercept by bin; fixed slopes = within-PCs
        endog = self.df['y']
        exog = sm.add_constant(self.df[[c for c in X_within.columns]])   # fixed effects
        exog_re = sm.add_constant(self.df[[c for c in X_within.columns]]) # random effects
        md = MixedLM(endog, exog, groups=self.df['bin'], exog_re=exog_re)
        mdf = md.fit(reml=True)
        self.y_hat = mdf.fittedvalues
        self.e_i = endog - self.y_hat
        print(mdf.summary())

    def fit_xgboost(self):

        columns = [
            'MedInc',
            'HouseAge',
            'Latitude',
            'Longitude',
            '_log1pTransformed_AveRooms',
            '_log1pTransformed_AveBedrms',
            '_log1pTransformed_AveOccup',
            '_log1pTransformed_Population'
        ]

        X_xgb = self.X[columns]

        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_xgb, self.y)

        self.y_hat = model.predict(X_xgb)
        self.e_i = self.y.values - self.y_hat        

        print(f"R²: {r2_score(self.y, self.y_hat):.3f}")
        print(f"RMSE: {root_mean_squared_error(self.y, self.y_hat):.3f}")

    def california_housing(self):

        self.ch = fetch_california_housing()
        self.preprocess()
        self.plot_histograms_X()
        self.partial_residuals_plots('mlr_diagnostics_partial_residuals_california_housing.png')

        # run diagnostics
        self.partial_residuals_plots('mlr_diagnostics_partial_residuals_california_housing.png')
        self.rj_vs_ei('mlr_diagnostics_rj_vs_ei_california_housing.png')
        self.y_distribution(bins=np.arange(0, 5.25, 0.25), filename='mlr_y_distribution_california_housing.png')
        self.diagnose_collinearity(center=False)
        
        # modelling
        self.fit_xgboost()

        # diagnostics
        self.compute_residuals_stats()
        self.compute_diagnostics()
        self.plot_y_obs_vs_y_pred('mlr_y_obs_y_pred_california_housing.png')
        self.plot_diagnostics('mlr_diagnostis_california_housing.png')

    def example_mixed_effects_model(self):

        # y = X@beta + Z@u + epsilon
        # y_ij + (beta0 + u0j) + (beta1 + u1j)x_ij + epsilon_ij

        n = 100
        epsilon = np.random.normal(0, 1, n)
        X = np.random.normal(0, 5, n)
        X = sm.add_constant(X)
        g = np.random.choice([0, 1, 2], size=n)
        Z = pd.get_dummies(g, prefix='X3_').astype(int)
        beta = np.array([0.2, 0.4])
        u = np.array([-0.54, 0.2, 0.4])
        y = X @ beta + Z @ u + epsilon
        df = pd.DataFrame(np.column_stack((y, X[:, 1], g)), columns=['y', 'X1', 'X2']) 

        formula = "y ~  X1"
        model = smf.mixedlm(formula, df, groups=df['X2']).fit()
        y_hat = model.predict(df)
        e_i = y.values.ravel() - y_hat
        print(model.summary())
        print(model.random_effects)

        ymin, ymax = np.min([y, y_hat]), np.max([y, y_hat])
        fig, ax = plt.subplots(1, 2, figsize=(5, 3))
        ax[0].scatter(y_hat, e_i)
        ax[0].axhline(y=0, color='black', linestyle='--')
        ax[1].scatter(y, y_hat)
        ax[1].plot([ymin, ymax], [ymin, ymax], 'k--')
        plt.show()



if __name__ == '__main__':

    mlr = MLR()
    mlr.california_housing()
    #mlr.example_mixed_effects_model()
    #mlr.collinearity_2d_example()
    #mlr.collinearity_3d_example()
    #mlr.variance_inflation_colinear_X(True)
    #mlr.model_basic(1000, 300, 10, [10.3, 5.4, -6.78])
    #mlr.model_interaction('mlr_distr_ei_missing_interaction.png')
    #mlr.model_categorical_predictors('mlr_scatterplot_with_categorical.png', 'mlr_diagnostics_with_categorical.png')
    #mlr.model_interaction_categorical_predictors('mlr_scatterplot_categorical_interaction.png', 'mlr.diagnostics_categorical_interaction.png')
    #mlr.model_missing_nonlinear_term('mlr_diagnostics_missing_nonlinear_ok.png', 'mlr_diagnostics_missing_nonlinear_missing.png')
    #mlr.model_heteroscedastic('mlr_diagnostics_heteroscedastic.png')

