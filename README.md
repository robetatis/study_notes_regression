# Regression

## Ordinary least-squares - OLS

### Model and intuition

Model: $y = X\beta + \epsilon$, with $\epsilon \sim N(0, \sigma)$, $\beta= \langle\beta_0, \beta_1\rangle$ and $X$ a column vector with a column of ones on the left. The deterministic part is $X\beta$, which is a model for $E[Y|X]$. $\epsilon$ is the random deviation of each $y_i$ from $E[Y|X]$, and is what makes $y$ a random variable.

The modelling task in OLS is to estimate $\hat{\beta}$ and its variability. That gives us an estimate of $E[Y|X]$ and a way to compute confidence intervals. $\hat{\beta}$ is computed as (expected value):

$\hat{\beta} = (X^TX)^{-1}X^Ty$

and the variance of its sampling distribution $Var(\hat{\beta})$ is related to 1) the variance of the residuals $\sigma^2 = Var(\epsilon)$ and 2) $X^TX$ (so-called Gram matrix), which captures the degree to which the regressors line up in $n$-dimensional space ($n$ = no. observations). The formula is:

$Var(\hat{\beta}) = \sigma^2(X^TX)^{-1}$

Two notes on this equation: 
1. $\sigma^2$ is not observable, so we have to use the sample estimate $\text{RSE} = \hat{\sigma}^2 = \text{RSS}/(n-p-1)$, with $p$ = no. regressors (=1 in OLS) and $\text{RSS} = e^Te$, where $e$ = sample residuals (see below). $\text{RSE}$ = Residual Standard Error and $\text{RSS}$ = Residual Sum of Squares. We're allowed to use $\text{RSE}$ as an estimate of $\sigma^2$ becasue, _if the OLS assumptions hold_, $E(\hat{\sigma}^2) = E(\text{RSE}) = \sigma^2$
2. The entries in $X^TX$ are just the dot-products $x_j^Tx_j$ for the diagonal elements and $x_j^Tx_{-j}$ for the off-diagonal elements. The former are related to the variance of regressor $x_j$ and the latter to the pairwise covariances between $x_j$ and each of the other regressors $x_{-j}$.

In general, larger samples with more spread $X$ have smaller variability; the mechanism is simply that larger samples encompass a larger part of the population and hence have better chances of representing it well. In turn, more spread in $X$ implies more chances to capture the way $y$ varies over $X$. Conversely, a narrow interval of $X$ doesn't allow that and therefore leads to more variable (i.e., more uncertain) estimates of $\hat{\beta}$.

Note: these formulas are only valid if $Cov(X, \epsilon) = 0$. The intuition here is that if $\epsilon$ is related to the regressors $X$, these contain information about the part of $y$ that is supposed to be only noise, i.e., there's still variance in $\epsilon$ that is linked to $X$. It's also possible that an omitted variable is moving with $X$ and thus affecting $y$ indirectly. This then causes $X$ and $\epsilon$ to be related. 

Since $Cov(X, \epsilon) = E(X\epsilon | X) - E(X)E(\epsilon | X) = XE(\epsilon) - E(X)E(\epsilon)$, the only way for $X$ and $\epsilon$ have zero covariance is if $E(\epsilon)=0$, which we achieve by correctly specifying the model (accounting for non-linearities and avoiding omitted variable bias). It's also worth noting that, with multiple regressors, checking which $X$ is responsible for $Cov(X, \epsilon) \neq 0$ helps determining which regressor drives the residuals.

### Sums of squares and $R^2$

After finding $\hat{\beta}$ we can compute $\hat{y}$, which is an estimate of $E(y\mid X)$. We can then also compute sample residuals $e_i = y_i - \hat{y}_i$. With $e_i$ we can compute sample estimates for the distribution of $\epsilon$, which we obviously don't have so we're forced to use $e_i$'s distribution as a proxy. A first measure of interest is $\sigma^2$, i.e., the variance of $\epsilon$'s distribution.

We estimate this quantity with the **residual standard error** $RSE = \sqrt{\text{RSS}/(n-2)}$, where the **residual sum of squares** $\text{RSS} = \sum_{i=1}^n (y_i - \hat{y}_i)^2$ and $n-2$ are the degrees of freedom when computing $\text{RSE}$ in OLS. We lose 2 degrees of freedom becasue we need to estimate two parameters - $\hat{\beta}_0$ and $\hat{\beta}_1$ to calculate $\text{RSE}$; in MLR (see below) we lose $p + 1$ degrees of freedom ($p = no. regressors + 1 for the intercept).

These sums of squares can be used to compute the % of explained variance in the regression, the $R^2$ statistic. $R^2 = 1 - \text{RSS}/\text{TSS}$, where the **total sum of squares** $\text{TSS} = \sum_{i=1}^n (y_i - \bar{y})$. This statistic measures the **linear** association between $X$ and $y$. In bivariate regression, the $R^2$ statistic is equal to the square of the _correlation coefficient_ $r^2$, which is equal to $Cov(X,y)/(\sqrt{Var(X)Var(y)})$. This is **not** true for multivariate settings.

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

Having multiple regressors opens up a host of possibilities, good and bad. Good because the world is multidimensional and including multiple regressors accounts for this; bad becasue regressors could be non-independent, we could be missing the right regressors, and we are forced to select a model, not to mention the problems with wide $X$ matrices, i.e., many features, few observations, and 'the curse of dimensionality'. More on all of these below.

In order to find $\hat{\beta}$ in $\hat{\beta} = (X^TX)^{-1}X^Ty$, **$X^TX$ must be invertible**, which requires the columns of $X$ to be _linearly indepenent_ (i.e., $X$ must have full column rank). The formula for $Var(\hat{\beta})$ is the same as in OLS (OLS is just a special case of MLR).

Intuitively, each entry $\hat{\beta}_j$ in **$\hat{\beta}$** is the slope of the relationship between $X_j$ and $E(y|X)$ _holding all other predictors fixed_. That is, $\hat{\beta}_j$ measures the **true** effect of $X_j$, or, in other words, it isolates the effect of $X_j$ by measuring it **in the presence of all other factors**. Indeed, the entries in vector $\hat{\beta}$ are 

$\hat{\beta}_j = \frac{r_j^Ty}{r_j^Tr_j}$

This is the **partial covariance formula**, and $r_j$ are the residuals of regressing $X_j$ against all other regressors $X_{-j}$. $r_j^Ty$ measures how much variation in $r_j$ lines up with variation in $y$, and dividing by $r_j^Tr_j$ sets that quantity in relation to the total variation in $r_j$.

The intuition behind partial covariance is: with multiple regressors, we can't directly use $X_j^Ty/(X_j^TX_j)$ to capture how much $X_j$ varies together with $y$. The reason is that the other regressors ($X_-j$) could also influence $X_j$. That means **what we need to look at is the covariance of $y$ with the part of $X_j$ that is not explained by the other regressors** - this is what 'the effect of $X_j$ holding all other regressors fixed' actually means. 

The part of $X_j$ not explained by the other regressors is captured by the residuals of regressing $X_j$ on all other regressors. $r_j$ can be understood as the 'isolated' variability of $X_j$, or as 'what remains' of $X_j$ after removing the influence of all other regressors. The covariance of that 'cleaned' $X_j$ with $y$ then gives the true, isolated effect of $X_j$ on $y$ in the presence of all other regressors. In terms of vectors and spaces, $r_j$ is always orthogonal to $X_{-j}$. Remember that a regression simply finds the projection of $y$ onto the column space of $X$, and the vector of residuals is the orthogonal distance between $y$ and that projection.

### Omitted variable bias

When we omit relevant regressors we are 'burying' the effect of the factors that are actually relevant in the $\hat{\beta}_j$ of the factors we do include. This happens when the omitted regressors meet two conditions: 1) the omitted regressors correlate with $y$ **and** 2) the omitted regressors correlate with the included regressors. Formally: 

Suppose the true model is $y = X\beta + \epsilon$. Say $X$ is of shape (n, m), but we only include $k$ regressors ($k < m$), i.e., we leave $m-k$ regressors out of the model. We can say $X$ is made up of two parts (included and omitted columns), so we can write it as $X = [X_1 \ X_2]$, with $X_1$ = the included regressors and $X_2$ = the omitted regressors. Each of these is just a matrix, one with the included columns ($X_1$) and another with the omitted ($X_2$) columns of $X$.

With this split, we can write the true model as $y = X_1\beta_1 + X_2\beta_2 + \epsilon$, where we've also split $\beta$ into an included part $\beta_1$ and an omitted part $\beta_2$. This implies $X\beta = X_1\beta_1 + X_2\beta_2$.

Then our incomplete model is $y = X_1\alpha + \delta$, and $\hat{y} = \hat{\alpha}X_1$, so $\hat{\alpha}$ is the regression coefficient we estimate when we only include the subset of regressors $X_1$. Now, by definition

$\hat{\alpha} = (X_1^TX_1)^{-1}X_1^Ty$. If we substitute in the true model for $y$, we can see how the missing $X_2$ influences $\hat{\alpha}$:

$\hat{\alpha} = (X_1^TX_1)^{-1}X_1^T(X_1\beta_1 + X_2\beta_2 + \epsilon)$. If we expand this

$\hat{\alpha} = \beta_1[(X_1^TX_1)^{-1}X_1^TX_1] + (X_1^TX_1)^{-1}X_1^TX_2\beta_2 + (X_1^TX_1)^{-1}X_1^T\epsilon$ (the 1st term in square brackets is $I$, the identity matrix). This simplifies to:

$\hat{\alpha} = \beta_1 + (X_1^TX_1)^{-1}X_1^TX_2\beta_2 + (X_1^TX_1)^{-1}X_1^T\epsilon$

If we take the expected value given $X_1$, $X_2$  on both sides:

$E(\hat{\alpha} | X_1, X_2) = \beta_1 + E[(X_1^TX_1)^{-1}X_1^TX_2\beta_2 | X_1, X_2] + E[(X_1^TX_1)^{-1}X_1^T\epsilon | X_1, X_2]$. Since $E(\epsilon | X_1, X_2)=0$,

$E(\hat{\alpha} | X_1, X_2) = \beta_1 + (X_1^TX_1)^{-1}X_1^TX_2\beta_2$

This result indicates that the expected value of $\hat{\alpha}$ is $\beta_1$ plus a **bias** (remember that bias = how much the expected value of our estimate deviates from the true value) that depends on 1) $\beta_2$ - the true effect of the omitted variable on $y$ and 2) $X_1^TX_2$ - how much the included and omitted regressors covary.

$(X_1^TX_1)^{-1}X_1^TX_2$ -> this term has two parts: $X_1^TX_2$ measures how strongly the included ($X_1$) and omitted ($X_2$) regressors covary, and $(X_1^TX_1)^{-1}$ acts as a 'distributor' of the previous quantity. Namely, this factor 'allocates' bias to the entries of $\hat{\alpha}$ based on how redundant the columns of $X_1$ are. Columns of $X_1$ that can be predicted from the others (collinear regressors) get more bias (due to the way entries in $(X_1^TX_1)^{-1}$ capture $Var(x_1,j)$ vs. $Cov(x_1,j, x_{1,-j})$ -> the variance and covariance of the included regressors).

In summary, if we omit variables that are possitively associated with $y$ and with $X_1$ (the included regressors) we will have inflated coefficients for the regressors we do include, since they represent the true value $\beta_1$ plus the bias $(X_1^TX_1)^{-1}X_1^TX_2\beta_2$. If the omitted variables are negatively associated with $y$ and with the included regressors, coefficient estimates will be smaller because the true effect will be dampened by the bias. In cases where the omitted variables have a positive (negative) relation with $y$ and a negative (positive) relation with $X_1$, the net effect depends on whether $\beta_2$ or $X_1^TX_2$ is larger.

### Hypothesis test

In MLR, we test $H_0: \beta_j = 0$ vs. $H_A:$ at least one $\beta_j \neq 0$, and the test statistic is 

$F = \frac{(\text{TSS} - \text{RSS})/p}{\text{RSS}/(n-p-1)}$.

What $F$ does is set the model effects ($TSS - RSS$) in relation to the residuals. If there _is_ a relationship between $X$ and $y$, $\text{RSS}$ will be smaller and $\text{TSS} - \text{RSS}$  will be larger, making $F > 1$. $F follows an F-distribution. No details here.

If H0 is true, the data-generating process is $y = \beta_0\text{1} + \epsilon$

RSS, TSS, are sample statistics because they come from a sample. duhh


### The curse of dimensionality


### Sparsity and collinearity




### Collinearity

inflated variance of beta