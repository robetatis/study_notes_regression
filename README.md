# Regression

## OLS

Model: $y = \beta_0 + \beta_1x + \epsilon$, with $\epsilon \sim N(\mu, \sigma)$

Deterministic part is $\beta_0 + \beta_1x$, which is a model for $E[Y|X]$

$\epsilon$ is the random deviation of each $y_i$ from $E[Y|X]$, and is what makes $y$ a random variable.

The modelling task in OLS is to estimate $\hat{\beta}$ and its variability. That gives us an estimate of $E[Y|X]$ and a way to compute confidence intervals. Beta is computed as:

$\hat{\beta} = (X^TX)^{-1}X^Ty$

and its variability comes from the variance-covariance matrix of the sampling distribution of $\hat{\beta}$:

$Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}$

In general, larger samples with more spread $X$ have smaller variability; the mechanism is simply that larger samples encompass a larger part of the population and hence have better chances of representing it well. In turn, more spread in $X$ implies more chances to capture the way $y$ varies over $X$. Conversely, a narrow interval of $X$ doesn't allow that and therefore leads to more variable (i.e., more uncertain) estimates of $\hat{\beta}$.

Note: these formulas are only valid if $Cov(X, \epsilon) = 0$. The intuition here is that if $\epsilon$ is related to the regressors $X$, these contain information about the part of $y$ that is supposed to be only noise, i.e., there's still variance in $\epsilon$ that is linked to $X$. It's also possible that an ommitted variable is moving with $X$ and thus affecting $y$ indirectly. This then causes $X$ and $\epsilon$ to be related. 

Since $Cov(X, \epsilon) = E(X\epsilon | X) - E(X)E(\epsilon | X) = E(X)E(\epsilon) - E(X)E(\epsilon)$, the only way $X$ and $\epsilon$ have zero covariance is if $E(\epsilon)=0$, which we achieve by correctly specifying the model (accounting for non-linearities and avoiding omitted variable bias). It's also worth noting that, with multiple regressors, checking which $X$ is responsible for $Cov(X, \epsilon) \neq 0$ helps determining which regressor drives the residuals.

### Sums of squares and $R^2$

After finding $\hat{\beta}$ we can compute $\hat{y}$, which is an estimate of $E(y\mid X)$. We can then also compute sample residuals $e_i = y_i - \hat{y}_i$. With $e_i$ we can compute sample estimates for the distribution of $\epsilon$, which we obviously don't have so we're forced to use $e_i$'s distribution as a proxy. A first measure of interest is $\sigma$, i.e., the variance of $\epsilon$'s distribution. We estimate this quantity with the $RSE = \sqrt{RSS/(n-2)}$


 by first calculating the **residual sum of squares** $\mathrm{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$ and then computing the sample **residual standard error** . Here, $n-2$ is used as denominator because in OLS we estimate two parameters ($\hat{\beta}_0$ and $\hat{\beta}_1$).

These sums of squares can be used to compute the % of explained variance in the regression, the $R^2$ statistic. $R^2 = 1 - RSS/TSS$, where the **total sum of squares** $TSS = \sum_{i=1}^n (y_i - \bar{y})$. This statistic measures the **linear** association between $X$ and $y$. In bivariate regression, the $R^2$ statistic is equal to the square of the _correlation coefficient_ $r^2$, which is equal to $Cov(X,y)/(\sqrt{Var(X)Var(y)})$. This is **not** true for multivariate settings.

Informally, looking at the size of RSS relative to the mean of $y$ is a way to assess how big unexplained variation is relative to the overall size of the response.

### OLS diagnostics

Before interpreting $\hat{\beta}_0$ and $\hat{\beta}_1$ and their statistical and business significance, we must check whether the linear model fit is usable. Several steps here:

1. Normality of residuals: Look at histogram of $e$ (sample residuals, since the population residuals require the entire population, which we obviously don't have). Formal tests: **Omnibus**, **Jarque-Bera**. Both have $H_0$: Distribution of $\epsilon_i$ is normal. However, given the Central Limit Theorem, normality of residuals is less critical with larger sample sizes ($n > 30$)

2. Autocorrelation of residuals: Look at fitted values vs. $e$. A random pattern suggests (however does not prove!) true linearity between $X$ and $y$. Any trend means systematic over/underpredicting depending on $y$; for instance, small values of $y$ overpredicted and large values of $y$ underpredicted -> we're likely missing a quadratic term -> model is **misspecified**. There can of course be other trends in fitted vals. vs. residuals: points along bands -> missing categorical variable, points funnel-shaped -> heteroscedasticity, etc. Formal test: **Ljung-Box**. Tests $H_0$: Zero residual autocorrelation.

3. Fitted values vs studentized residuals: This plot allows finding outliers and patterns in residuals (the same as above). The fact that residuals are studentized makes it easier to determine outliers (when the $t$ value is higher than, say 3). 

4. Leverage v studentized residuals: Outliers are bad, but outliers with high leverage are worse. Points with high leverage have a strong influence on $\hat{\beta}$ and can therefore distort relations since the regression would be heavily biased by only a few datapoints. Outliers with low leverage are less problematic, although they increase $RMSE$ and therefore lead to lower $R^2$.

## MLR

### Ommitted variable bias

What happens if the true model is $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon$, but we regress $y = \alpha_0 + \alpha_1x_1 + u$? To find out, we can look at how the estimated parameter $\hat{\alpha}_1$ is related to $ \beta_1$, i.e., how the true $\beta_1$ is contained in the estimated $\hat{\alpha_1}$. In OLS, 

$\hat{\alpha_1} = Cov(x_1, y)/Var(x_1)$ 

Since the true $y$ is $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon$, then

$\hat{\alpha_1} = Cov(x_1, \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon)/Var(x_1)$. 

Since covariance is linear, we get 

$\hat{\alpha_1} = (\beta_1Cov(x_1, x_1) + \beta_2Cov(x_1,x_2) + Cov(x_1, \epsilon))/Var(x_1)$. 

Since $Cov(x_1, x_1) = Var(x_1)$ and $Cov(x_1, \epsilon) = 0$ (OLS assumption!), we get 

$\hat{\alpha_1} = \beta_1Var(x_1) + \beta_2\frac{Cov(x_1,x_2)}{Var(x_1)}$

Simplifying,

$\hat{\alpha_1} = \beta_1 + \beta_2\frac{Cov(x_1, x_2)}{Var(x_1)}$. 

That means the ommitted variable $x_2$ biases the estimated $\hat{\alpha}$ by the amount $\beta_2Cov(x_1, x_2)/Var(x_1)$. This same logic extends to more regressors.

Intuitively this means ommitting a variable that is related to $y$ ($\beta_2 \neq 0$) and to $x_1$ ($Cov(x_1, x_2) \neq 0$) results in a biased regression coefficient, which may be smaller or larger than the true effect of $x_1$, i.e., $\beta_1$.

exampl
           ols     mlr
tv        0.0475  0.046  
radio     0.203   0.189  
newspaper 0.055   -0.001

similar ols and mlr coefficients for tv and radio -> tv and radio are independent.
newspaper drops to non-significance -> wasn't independent, was confounded with either tv or radio



### Collinearity

inflated variance of beta