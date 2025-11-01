# multi-level linear models (MLLMs)
# --------------------------------------------------------------------------

# intro
# -----------------------------------
# - used when data are naturally grouped (patients in hospitals, employees in companies/branches/departments, students in schools, etc.)
# - observations being grouped/nested means datapoints ARE NOT INDEPENDENT!!
# - in classical models, there's only one global slope and global intercepts for all observations
# - that means those parameters are *fixed*
# - this also implies that the relation between Y and X is the same across all observations, which, with nested/grouped data, is likely not true
# - another way to say this is that each value of Y is drawn from a single trend + random noise, i.e., we ignore groups
# - in a classical model, we could account for groups by using dummy variables and model their effect as fixed effects
# - however, this doesn't scale when we have many groups
# - in mllms we allow each group to have its own parameters (intercept and/or slope) and assume these params come from a common distribution
# - that is to say, in mllms, PARAMETERS HAVE DISTRIBUTIONS, AND ARE THEREFORE NOT FIXED NUMBERS
# - this way we acknowledge in the modelling that the data-generating process has structure across groups.
# - the distribution of the groups' parameters has its own parameters -> hyperparameters.
# - for instance, group intercepts alpha_j ~ N(miu_alpha, sigma_alpha^2), here,  miu_alpha and sigma_alpha^2 are the mean and variance of the distribution
#   from which we draw the group intercepts
# - there are 2 extremes when handling naturally nested data:
#   - pooled model - single model for all data -> biased estimates, ignores/wastes structure given by between-group differences
#   - unpooled model - one separate model per group -> needs lots of datapoints per group, small groups have noisy estimates
# - mllms are a middle ground: partial pooling -> each group has its own params BUT they all come from a single distribution
# - in partial pooling, groups with lots of data behave as if they had their own separate regression, and groups with little data 'shrink' towards the global params
# - 'shrinkage':
#   - there's a global effect (global intercept, global slope) and each group's params are deviations (upwards or downwards) from the global params
#   - groups with little data see their effects 'shrink' towards zero, so these groups's effects are closer to the global trend


# model
# -----------------------------------
# two-level example, only one regressor x:
# y_ij = beta_0j + beta_1j*x_ij + epsilon_ij, epsilon_ij ~ N(0, sigma^2)
# beta_0j = gamma_00 + u_0j, u_0j ~ N(0, tau_0^2)
# beta_1j = gamma_10 + u_1j, u_1j ~ N(0, tau_1^2)
# y_ij = outcome i of group j
# x_ij = value of predictor for observation i of group j
# epsilon_ij = corresponding residual
# beta_0j, beta_1j = intercept and slope of group j
# gamma_00, gamma_10 = global intercept and slope
# u_0j, u_1j = group-level deviation from global intercept and slope
# tau_0^2, tau_1^2 = variance of random effects - hyperparameters

# manual example with moment of methods
# -----------------------------------
# y_ij = beta_0 + u_0j + epsilon_ij, epsilon_ij ~ N(0, sigma^2), u_0j ~ N(0, tau_0^2)
# we must estimate beta_0, u_0j, sigma^2 and tau_0^2
# Best Linear Unbiased Predictor for u_0j is:
# u_0j_hat = tau_hat^2/(tau_hat^2 + sigma^2_hat/n_j)*(y_j_bar - y_bar)
# beta_0_hat = y_bar
# j=1, n_1 = 2, y = [8, 9]
# j=2, n_2 = 3, y = [2, 4, 3]
# using method of moments
# group means:
#   y_1_bar = (8+9)/2 = 8.5
#   y_2_bar = (2+4+3)/3 = 3
# overall mean =(8+9+2+4+3)/5 = 5.2
# SS_within, group 1 = (8-8.5)^2 + (9-8.5)^2 = 0.5
# SS_within, group 2 = (2-3)^2 + (4-3)^2 + (3-3)^2 = 2
# SS_within = 0.5 + 2 = 2.5
# SS_between = 2*(8.5-5.2)^2 + 3*(3-5.2)^2 = 36.3
# MS_within = 2.5/(N-g) = 2.5/(5-2) = 0.833
# MS_between = 36.3/(g-1) = 36.3/(2-1) = 36.3
# tau_0^2_hat = (MS_between - MS_within)/n_bar = (36.3 - 0.833)/(2.5) = 14.18
# sigma^2_hat = MS_within = 0.833
# w_j = tau_hat^2/(tau_hat^2 + sigma^2_hat/n_j)
#   w_1 = 14.18/(14.18 + 0.833/2) = 0.971
#   w_2 = 14.18/(14.18 + 0.833/3) = 0.981
# u_0j_hat = w_j*(y_j_bar - y_bar)
#   u_01_hat = 0.971*(8.5 - 5.2) = 3.2
#   u_02_hat = 0.981*(3 - 5.2) = -2.16
# predicted group means:
#   y_bar + u_01_hat = 5.2 + 3.2 = 8.4
#   y_bar + u_02_hat = 5.2 - 2.16 = 3.04
# since tau_0_hat^2 (14.18) is large relative to sigma^2_hat (0.833), these are close to the raw y_j_bars: 8.5, 3


# sleepstudy dataset
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import omni_normtest, jarque_bera
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor
import matplotlib.pyplot as plt

class Model:

    def __init__(self):
        self.get_data()

    def get_data(self):

        data_r = sm.datasets.get_rdataset("sleepstudy", "lme4").data
        self.df_raw = pd.DataFrame(data_r)
        self.df = pd.get_dummies(self.df_raw, columns=['Subject'], drop_first=True)

        self.X = self.df.loc[:, self.df.columns != "Reaction"]
        self.X = self.X.astype(float)
        self.y = self.df['Reaction']

    def ols_compute_residuals_stats(self):
        self.e_i_mean = np.mean(self.e_i)
        self.e_i_sd = np.sqrt(np.var(self.e_i, ddof=2)) # this is rse (the residuals' standard error)
        self.e_i_x = np.linspace(self.e_i.min(), self.e_i.max(), 300)
        self.e_i_pdf = scipy.stats.norm.pdf(self.e_i_x, loc=self.e_i_mean, scale=self.e_i_sd)

    def ols_compute_vif(self):
        self.vif = pd.DataFrame()
        self.vif['feature'] = self.X.columns
        self.vif['vif'] = [variance_inflation_factor(self.X.values, i)  for i in range(self.X.shape[1])]
        print(self.vif)

    def ols_compute_diagnostics(self):

        self.omnibus_stat, self.omnibus_p = omni_normtest(self.e_i) 
        self.jb_stat, self.jb_p, self.skew, self.kurtosis = jarque_bera(self.e_i)
        lb = acorr_ljungbox(self.e_i, lags=[1], return_df=True)
        self.lb_stat, self.lb_pvalue = lb['lb_stat'].values[0], lb['lb_pvalue'].values[0]

        XtX_inv = np.linalg.inv(self.X.T @ self.X)
        self.leverage = np.einsum('ij,jk,ik->i', self.X, XtX_inv, self.X)
        mse = np.mean(self.e_i**2)
        self.studentized_resid = self.e_i / np.sqrt(mse * (1 - self.leverage))

        self.ols_compute_vif()

    def ols_plot_diagnostics(self, filename):
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

    def ols_plot_y_obs_vs_y_pred(self, filename):

        ymin = np.min([self.y, self.y_hat])
        ymax = np.max([self.y, self.y_hat])

        fig, ax = plt.subplots()
        ax.scatter(self.y, self.y_hat, alpha=0.3)
        ax.plot([ymin, ymax], [ymin, ymax], 'k--')
        ax.set_ylabel('y pred')
        ax.set_xlabel('y obs')
        plt.savefig(filename)

    def ols_run(self):
        all_columns = " + ".join(self.df.loc[:, self.df.columns != "Reaction"])

        formula = f"Reaction ~ {all_columns}"
        self.model = smf.ols(formula, data=self.df).fit()
        self.y_hat = self.model.fittedvalues
        self.e_i = self.df['Reaction'].values.ravel() - self.y_hat
        print(self.model.summary())

        self.ols_compute_residuals_stats()
        self.ols_compute_diagnostics()
        self.ols_plot_diagnostics('output/mllm_ols_diagnostics_sleepstudy.png')

    def mllm_plot_results(self):
        random_effects = self.results.random_effects
        plt.figure(figsize=(10,6))
        for subj in self.df_raw['Subject'].unique():
            subj_data = self.df_raw[self.df_raw['Subject'] == subj]

            u0 = random_effects[subj]['Subject']
            u1 = random_effects[subj]['Days']
            
            y_pred = (self.results.fe_params['Intercept'] + u0) + (self.results.fe_params['Days'] + u1) * subj_data['Days']
            
            plt.plot(subj_data['Days'], y_pred, label=f'Subject {subj}')
            plt.scatter(subj_data['Days'], subj_data['Reaction'], color='black', s=10)

        plt.xlabel('Days of Sleep Deprivation')
        plt.ylabel('Reaction Time')
        plt.title('Subject-Specific Lines (Random Intercept + Slope)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout()
        plt.savefig('output/mllm_results_sleepstudy.png')

    def mllm_plot_diagnostics(self):

        fitted = self.results.fittedvalues
        resid = self.results.resid

        # residuals vs Fitted
        plt.figure(figsize=(6,4))
        plt.scatter(fitted, resid, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residuals vs Fitted Values")
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.savefig('output/mllm_diagnostics_sleepstudy_fitted_vs_e_i.png')

        # qq-plot of residuals
        plt.figure(figsize=(5,5))
        stats.probplot(resid, dist="norm", plot=plt)
        plt.title("QQ-Plot of Conditional Residuals")
        plt.savefig('output/mllm_diagnostics_sleepstudy_qq_e_i.png')

        # qq-plots of random effects
        # -------------------------
        random_effects = pd.DataFrame(self.results.random_effects).T  # subjects x effects

        for col in random_effects.columns:
            plt.figure(figsize=(5,5))
            stats.probplot(random_effects[col], dist="norm", plot=plt)
            plt.title(f"QQ-Plot of Random Effect: {col}")
            plt.savefig(f'output/mllm_diagnostics_sleepstudy_qq_{col}.png')

        # variance by group (optional)
        # -------------------------
        group_sd = (
            pd.DataFrame({'Subject': self.results.model.groups, 'Resid': resid})
            .groupby('Subject')['Resid'].std()
        )
        plt.figure(figsize=(6,4))
        plt.bar(group_sd.index, group_sd.values)
        plt.title("Residual SD per Subject")
        plt.xlabel("Subject")
        plt.ylabel("Within-group residual SD")
        plt.savefig('output/mllm_diagnostics_sleepstudy_std_by_group.png')

    def mllm_run(self):
        # random intercept-random slope model: 
        # y_ij = (u_00 + u_0j) + (u_10 + u_1j)*Days + epsilon_ij, epsilon_ij ~ N(0, sigma^2)

        model = smf.mixedlm(
            'Reaction ~ Days', # fixed effects
            self.df_raw,        
            groups='Subject',   
            re_formula='1 + Days' # random effects 1=intercept, Days=slope for this regressor
        )
        self.results = model.fit(reml=True)
        print(self.results.summary())

        print(f"sd(y) = {self.df_raw['Reaction'].std():.2f}")
        print(f'sd(e_i) = {np.sqrt(self.results.scale):.2f}')
        print(f'sd(u_0j) = {np.sqrt(self.results.cov_re.loc['Subject', 'Subject']):.2f}')
        print(f'sd(u_1j) = {np.sqrt(self.results.cov_re.loc['Days', 'Days']):.2f}')

        random_effects = pd.DataFrame(self.results.random_effects)
        print(random_effects.T)

        self.mllm_plot_results()
        self.mllm_plot_diagnostics()


if __name__ == '__main__':
    model = Model()
    #model.ols_run()
    model.mllm_run()