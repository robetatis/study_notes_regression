# GLMs
# --------------------------------------------------------------------------------
# in linear regression, we say Y_i = X_i^T*beta + epsilon_i, with epsilon_i ~ N(0, sigma^2) being the random element
# epsilon_i being random implies Y_i is also random, Y_i ~ N(X_i^T*beta, sigma^2)
# in GLMs we completely reframe the problem:
# 1. we directly say the response is random: Y_i ~ Distribution(mean = miu_i, dispersion = phi)
# 2. we create a linear predictor that combines the regressors: eta_i = X_i^T*beta
# 3. since eta_i can be in the range (-infinity, infinity), we use a link function that maps it to the scale of miu_i
#    eta_i = g(miu_i); miu_i = g^-1(eta_i)
# links:
#   - logit: log(miu_i/(1 - miu_i)) = X_i^T*beta, with miu_i = p_i -> logistic regression
#   - log: log(miu_i) = X_i^T*beta, with miu_i the mean of a Poisson distr. -> Poisson regression


# logistic regression
# --------------------------------------------------------------------------------
# Y_i {0, 1}. we model P(Y_i = 1 | X_i)
# Y_i ~ Bernoulli(p_i), E[Y_i] = p_i, Var[Y_i] = p_i*(1-p_i) -> as in all GLMs, variance depends on mean, and the model is inherently heteroscedastic
# logistic/logit link: g(p) = log(p/(1-p)) -> log-odds of success, were p/(1-p) = odds
# model: g(p_i) = log(p_i/(1-p_i)) = X_i^T*beta ; p_i = 1/(1 + exp(-X_i*beta))
# beta_j = increase in log-odds of success per unit increase in X_j holding all other X_j constant
# exp(beta_j) is therefore multiplicative increase in odds -> odds ratio
# we maximize L(beta) = prod[p_i^(y_i)*(1-p_i)^(1-y_i)], or equivalently:
# we minimize (because optim. usually minimizes instead of maximizing) -log(L(beta)) = -l(beta) = -sum[p_i^(y_i)*(1-p_i)^(1-y_i)]

import numpy as np
import pandas as pd
import kagglehub
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class GLM:
    
    def logistic_regression(self):

        # get data
        path = kagglehub.dataset_download("pavansubhasht/ibm-hr-analytics-attrition-dataset")
        path = Path(path) / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
        self.data_employee_attrition = pd.read_csv(
            path
        )

        # preprocess
        y = (self.data_employee_attrition['Attrition'] == 'Yes').astype(int)
        X = self.data_employee_attrition[['Age', 'MonthlyIncome', 'JobSatisfaction', 'OverTime']]
        X = pd.get_dummies(X, columns=['OverTime'], drop_first=True) 
        X = sm.add_constant(X)
        X = X.astype(float)

        # fit model
        model = sm.Logit(y, X).fit(disp=False)
        print(model.summary())

        # show confidence intervals of beta_j
        params = model.params
        conf = model.conf_int()
        conf['OR'] = np.exp(params)
        conf['2.5%'] = np.exp(conf[0])
        conf['97.5%'] = np.exp(conf[1])
        print(conf)

        # goodness of fit: log-likelihood ratio test
        lr_test = model.llr 
        lr_pvalue = model.llr_pvalue
        print(f"LR test = {lr_test:.2f}, p = {lr_pvalue:.4f}")

        # predictive performance
        pred_probs = model.predict(X)
        pred_class = (pred_probs > 0.5).astype(int)

        fpr, tpr, _ = roc_curve(y, pred_probs)
        auc = roc_auc_score(y, pred_probs)

        print(confusion_matrix(y, pred_class))
        print(classification_report(y, pred_class))

        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve (AUC = {auc:.2f})')
        plt.savefig('output/glm_logistic_regression_auc.png')

        # Age = -0.037 -> odds = exp(-0.037) = 0.96. next year's odds of leaving are 4% smaller
        # MonthlyIncome x1000 = -9.94e-2 -> odds = exp(-9.94e-2 * 1000) = 0.905. every 1000k more in income/month make attrition 10% less likely
        # JobSatisfaction = -0.3 -> odds = exp(-0.301) = 0.74. every step in job satisfactoin makes attrition 25% less likely
        # overtime = 1.4 -> odds = exp(1.4) = 4.05. overtime makes attrition 4 times more likey



if __name__ == '__main__':
    glm = GLM()
    glm.logistic_regression()



#iteratively-reweighted least squares
#binary response, poisson regression
#quasi-poisson, negative binomial
#zero-inflated count regression

