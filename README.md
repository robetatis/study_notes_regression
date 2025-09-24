# Regression

## Ordinary least-squares - OLS

### Model and intuition

Model: $y = X\beta + \epsilon$, with $\epsilon \sim N(0, \sigma)$, $\beta= \langle\beta_0, \beta_1\rangle$ and $X$ a column vector. The deterministic part is $X\beta$, which is a model for $E[Y|X]$. $\epsilon$ is the random deviation of each $y_i$ from $E[Y|X]$, and is what makes $y$ a random variable.

The modelling task in OLS is to estimate $\hat{\beta}$ and its variability. That gives us an estimate of $E[Y|X]$ and a way to compute confidence intervals. $\hat{\beta}$ is computed as:

$\hat{\beta} = (X^TX)^{-1}X^Ty$

and its variability comes from the variance-covariance matrix of the sampling distribution of $\hat{\beta}$:

$Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}$

In general, larger samples with more spread $X$ have smaller variability; the mechanism is simply that larger samples encompass a larger part of the population and hence have better chances of representing it well. In turn, more spread in $X$ implies more chances to capture the way $y$ varies over $X$. Conversely, a narrow interval of $X$ doesn't allow that and therefore leads to more variable (i.e., more uncertain) estimates of $\hat{\beta}$.

Note: these formulas are only valid if $Cov(X, \epsilon) = 0$. The intuition here is that if $\epsilon$ is related to the regressors $X$, these contain information about the part of $y$ that is supposed to be only noise, i.e., there's still variance in $\epsilon$ that is linked to $X$. It's also possible that an ommitted variable is moving with $X$ and thus affecting $y$ indirectly. This then causes $X$ and $\epsilon$ to be related. 

Since $Cov(X, \epsilon) = E(X\epsilon | X) - E(X)E(\epsilon | X) = XE(\epsilon) - E(X)E(\epsilon)$, the only way for $X$ and $\epsilon$ have zero covariance is if $E(\epsilon)=0$, which we achieve by correctly specifying the model (accounting for non-linearities and avoiding omitted variable bias). It's also worth noting that, with multiple regressors, checking which $X$ is responsible for $Cov(X, \epsilon) \neq 0$ helps determining which regressor drives the residuals.

### Sums of squares and $R^2$

After finding $\hat{\beta}$ we can compute $\hat{y}$, which is an estimate of $E(y\mid X)$. We can then also compute sample residuals $e_i = y_i - \hat{y}_i$. With $e_i$ we can compute sample estimates for the distribution of $\epsilon$, which we obviously don't have so we're forced to use $e_i$'s distribution as a proxy. A first measure of interest is $\sigma$, i.e., the variance of $\epsilon$'s distribution.

We estimate this quantity with the **residual standard error** $RSE = \sqrt{RSS/(n-2)}$, where the **residual sum of squares** $RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2$ and $n-2$ are the degrees of freedom when computing $RSE$ (we lose two degrees of freedom becasue we need to estimate two parameters - $\hat{\beta}_0$ and $\hat{\beta}_1$ to calculate $RSE$).

These sums of squares can be used to compute the % of explained variance in the regression, the $R^2$ statistic. $R^2 = 1 - RSS/TSS$, where the **total sum of squares** $TSS = \sum_{i=1}^n (y_i - \bar{y})$. This statistic measures the **linear** association between $X$ and $y$. In bivariate regression, the $R^2$ statistic is equal to the square of the _correlation coefficient_ $r^2$, which is equal to $Cov(X,y)/(\sqrt{Var(X)Var(y)})$. This is **not** true for multivariate settings.

Informally, looking at the size of RSS relative to the mean of $y$ is a way to assess how big unexplained variation is relative to the overall size of the response.

### OLS diagnostics

Before interpreting $\hat{\beta}_0$ and $\hat{\beta}_1$ and their statistical and business significance, we must check whether the linear model fit is usable. Several steps here:

1. Normality of residuals: Look at histogram of $e$ (sample residuals, since the population residuals require the entire population, which we obviously don't have). Formal tests: **Omnibus**, **Jarque-Bera**. Both have $H_0$: Distribution of $\epsilon_i$ is normal. However, given the Central Limit Theorem, normality of residuals is less critical with larger sample sizes ($n > 30$)

2. Autocorrelation of residuals: Look at fitted values vs. $e$. A random pattern suggests (however does not prove!) true linearity between $X$ and $y$. Any trend means systematic over/underpredicting depending on $y$; for instance, small values of $y$ overpredicted and large values of $y$ underpredicted -> we're likely missing a quadratic term -> model is **misspecified**. There can of course be other trends in fitted vals. vs. residuals: points along bands -> missing categorical variable, points funnel-shaped -> heteroscedasticity, etc. Formal test: **Ljung-Box**. Tests $H_0$: Zero residual autocorrelation.

3. Fitted values vs studentized residuals: This plot allows finding outliers and patterns in residuals (the same as above). The fact that residuals are studentized makes it easier to determine outliers (when the $t$ value is higher than, say 3). 

4. Leverage v studentized residuals: Outliers are bad, but outliers with high leverage are worse. Points with high leverage have a strong influence on $\hat{\beta}$ and can therefore distort relations since the regression would be heavily biased by only a few datapoints. Outliers with low leverage are less problematic, although they increase $RMSE$ and therefore lead to lower $R^2$.

## Multiple linear regression - MLR

### Model and intuition

Model is the same as above: $y = X\beta + \epsilon$, with $\epsilon \sim N(0, \sigma)$, but now $\beta$ is a vector of shape ($p$+1, 1), with $p$ = number of regressors (+1 for the intercept), and $X$ is no longer a column vector but a matrix of shape ($n$, $p+1$), i.e, $n$ observations and $p$ factors (again, +1 for the intercept, for which we add a column of ones as the first column of $X$).

Having multiple regressors opens up a host of possibilities, good and bad. Good because the world is multidimensional and including multiple regressors accounts for this; bad becasue regressors could be non-independent, we could be missing the right regressors, and we are forced to select a model, not to mention the problems with wide $X$s, i.e., many features, few observations, and 'the curse of dimensionality'. More on all of these below.

In order to find $\hat{\beta}$ in $\hat{\beta} = (X^TX)^{-1}X^Ty$, **$X^TX$ must be invertible**, which requires the columns of $X$ to be _linearly indepenent_ (i.e., $X$ must have full column rank). The formula for $Var(\hat{\beta})$ is the same as in OLS (OLS is just a special case of MLR).

Intuitively, each entry $\hat{\beta}_j$ in **$\hat{\beta}$** is the slope of the relationship between $X_j$ and $E(y|X)$ _holding all other predictors fixed_. That is, $\hat{\beta}_j$ measures the **true** effect of $X_j$, or, in other words, it isolates the effect of $X_j$ by measuring it **in the presence of all other factors**. Indeed, the entries in vector $\hat{\beta}$ are 

$\hat{\beta}_j = \frac{r_j^Ty}{r_j^Tr_j}$

This is the **partial covariance formula**, and $r_j$ are the residuals of regressing $X_j$ against all other regressors $X_{-j}$. $r_j^Ty$ measures how much variation in $r_j$ lines up with variation in $y$, and dividing by $r_j^Tr_j$ sets that quantity in relation to the total variation in $r_j$.

The intuition behind partial covariance is: with multiple regressors, we can't directly use $X_j^Ty/(X_j^TX_j)$ to capture how much $X_j$ varies together with $y$. The reason is that the other regressors ($X_-j$) could also influence $X_j$. That means **what we need to look at is the covariance of $y$ with the part of $X_j$ that is not explained by the other regressors** - this is what 'the effect of $X_j$ holding all other regressors fixed' actually means. The part of $X_j$ not explained by the other regressors is captured by the residuals of regressing $X_j$ on all other regressors. $r_j$ can be understood as an 'isolated' variability of $X_j$, or as $X_j$ after removing the influence of all other regressors. The covariance of that 'cleaned' $X_j$ with $y$ then gives the true, isolated effect of $X_j$ on $y$ in the presence of all other regressors. In terms of vectors and spaces, $r_j$ is always orthogonal to $X_{-j}$. Remember that a regression simply finds the projection of $y$ onto the column space of $X$, and the vector of residuals is the orthogonal distance between $y$ and the column space of $X$.

### Ommitted variable bias

Omitting relevant regressors runs the risk of false attribution. For instance, say a bivariate $y$-$X$ OLS gives a statstically significant $\hat{\beta}=0.3$, and then a MLR with the same $y$, the previous $X$ and two additional factors, gives non-significant $\hat{\beta}_1$ = 0.001 and siginificant $\hat{\beta}_2=0.51$ and $\hat{\beta}_3=0.32$. That would imply 1) that the bivariate OLS was 'burying' the effect of the factors that are actually relevant ($\hat{\beta}_2$ and $\hat{\beta}_3$) in $\hat{\beta}_1$, and 2) That the missing factors (2 and 3) are somehow related to the one we included in the OLS; the reason being that they exerted their influence on $y$ indirectly by affecting factor 1. These *interrelations amongst regressors are one of the main complexities of MLR*, and are formally referred to as **ommited variable bias**, i.e., what happens when we do not include relevant factors in our regression which affect $y$ and the factors we do include? In that case, those relevant factors act indirectly and lead to bias in the estimated $\hat{\beta}$.

This can be summarized as follows:

**************This needs to be changed to the version for multiple regressors. currently is for a single regressor*****************


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


### The curse of dimensionality


### Sparsity and collinearity




### Collinearity

inflated variance of beta