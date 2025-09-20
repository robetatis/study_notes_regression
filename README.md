# Regression

## OLS

Model: $y = \beta_0 + \beta_1x + \epsilon$, with $\epsilon \sim N(\mu, \sigma)$

Deterministic part is $\beta_0 + \beta_1x$, which is a model for $E[Y|X]$

$\epsilon$ is the random deviation of each $y_i$ from $E[Y|X]$, and is what makes $y$ a random variable.

variation upon replication. larger sample size -> smaller variation. formulas for sd(beta). smaller sd(beta) with larger spread of x sum(x_i - mean(x)). formulas only valid with uncorrelated epsilon, cor(epsilon,x)=0

rss relative to y mean is a way to assess how big unexplained variation is relative to the overall size of the response $y$

$rss = \sum_{i=1}^n(\hat{y}_i - y_i)^2$

$rse = \sqrt{\frac{1}{n-2}rss}$

normality of residuals -> look at histogram of epsilon.  
- omnibus, jarque-bera

residual autocorrelation -> look at fitted values vs. epsilon. random pattern suggests (does not prove) linearity since no trend in residuals. trend in residuals vs. fitted values means systematic over/underpredicting depending on y -> model misspecified (e.g., missing $x^2$). one could also see other trends in fitted vs. residuals: points along bands -> missing categorical variable, points funnel-shaped -> heteroscedasticity
- ljung box

epsilon correlated with x -> variable-specific misfit (which variable drives the residuals). if not zero, std(beta) formulas don't apply

fitted values vs studentized residuals
outliers: leverage v studentized residuals

r2 = (TSS - RSS) / TSS, % of explained variance

in ols this is equal to cov(x,y)/(std(x)std(y))

## MLR

### Ommitted variable bias

What happens if the true model is $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon$, but we regress $y = \alpha_0 + \alpha_1x_1 + u$. We can look at how the estimated parameter $\hat{\alpha}_1$ related to $ \beta_1$, i.e., how the true $\beta_1$ is contained in the estimated $\hat{\alpha_1}$. In OLS, 
$\hat{\alpha_1} = Cov(x_1, y)/Var(x_1)$, and since the true $y$ is $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon$, then $\hat{\alpha_1} = Cov(x_1, \beta_0 + \beta_1x_1 + \beta_2x_2 + \epsilon)/Var(x_1)$. Since covariance is linear, we get $\hat{\alpha_1} = (\beta_1Cov(x_1, x_1) + \beta_2Cov(x_1,x_2) + Cov(x_1, \epsilon))/Var(x_1)$. Since $Cov(x_1, x_1) = Var(x_1)$ and $Cov(x_1, \epsilon) = 0$ (OLS assumption!), $\hat{\alpha_1} = (\beta_1Var(x_1) + \beta_2Cov(x_1,x_2))/Var(x_1)$, i.e., $\hat{\alpha_1} = \beta_1 + \beta_2Cov(x_1, x_2)/Var(x_1)$. That means the ommitted variable $x_2$ biases the estimated $\hat{\alpha}$ by the amount $\beta_2Cov(x_1, x_2)/Var(x_1)$. This same logic extends to more regressors.

Intuitively this means ommitting a variable that is related to $y$ ($\beta_2 \neq 0$) and to $x_1$ ($Cov(x_1, x_2) \neq 0$) results in a biased regression coefficient, which may be smaller or larger than the true effect of $x_1$, i.e., $\beta_1$.

isn't it a contradiction that we're ommitting a variable and we're still assumming cov(x1, epsilon) = 0?


### Collinearity

inflated variance of beta