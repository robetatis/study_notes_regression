# Hierarchical models

Based on [UCLA STATS 205](https://www.youtube.com/watch?v=flEIC4_bt8c&list=PLAYxx7zX5F1O2HbRr4gORnscbM9EszYbK&index=1)

## Starting point: the multiple linear model

The response is $y = \langle y_1, y_2, ..., y_n \rangle^T$, which is continuous.

We have a **fixed (i.e., non-random)** design matrix $X$:

$$
X =
\begin{bmatrix}
1 & x_{11} & x_{12} & ... & x_{1(p-1)}\\
1 & x_{21} & x_{22} & ... & x_{2(p-1)}\\
... \\
1 & x_{n1} & x_{n2} & ... & x_{n(p-1)}\\
\end{bmatrix}
$$

Note that $X$ has $p-1$ predictors and a column of ones on the left for the intercept.

The data can also be denoted as $\{(x_i, y_i)\}_{i=1}^n$, where $x_i$ (a row of $X$) is in $\mathbb{R}^{p-1}$ and $y_i$ is in $\mathbb{R}$.

This setup, where we condition on $X$ and thereofore assume it's **fixed (i.e., non-random)**, is called the _fixed design problem_, where the modelling task is to find a function of $X$ that describes $E(y \mid X)$. Setups where $X$ is not assumed fixed give rise to the so-called _random design problem_.

The model for $y$ is $y = X\beta + \epsilon$. Here, $y$ is $n$ x 1, $X$ is $n$ x $p$, $\beta$ is $p$ x 1 and $\epsilon$ is $n$ x 1.

$\epsilon$ is an $n$-dimensional **random error vector** $\epsilon = \langle \epsilon_1, ... \epsilon_n \rangle$, where each $\epsilon_i$ is a random variable and we assume they are _iid_ with $E(\epsilon_i) = 0$. We can further assume that $\epsilon_i \sim N(0, \sigma^2)$, i.e., that the $\epsilon_i$ all all normally distributed with mean zero and variance $\sigma^2$ (_homoscedasticity_).

With this setup, the parameter set defining the model is $\theta = (\beta ,\sigma^2)$. $\theta$ can be estimated either via $L2$-norm (ordinary least squares OLS) or Maximum Likelihood Estimation MLE. OLS does not rely on any distributional assumptions for $\epsilon_i$, whereas MLE relies on $\epsilon_i \sim N(0, \sigma^2)$. Both estimation methods coincide under the previous assumption, i.e., $\hat{\beta}_{MLE} = \hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty$. The result is a $p$ x 1 vector $\hat{\beta}^T = \langle \hat{\beta}_0, \hat{\beta}_1, ..., \hat{\beta}_{p-1} \rangle$

### Bias and variance of $\hat{\beta}$

If _exogeneity_ holds, i.e., if $E(\epsilon \mid X) = 0$, then $E(\hat{\beta}) = \beta$. In this case, $\hat{\beta}$ is an unbiased estimator of $\beta$. $\hat{\beta}$ is a $p$ x 1 column vector. Note that if $E(\epsilon \mid X) = 0$, $\text{Cov}(\epsilon, X)$, which is equal to $\text{Cov}(\epsilon, X) = E(\epsilon X) - E(\epsilon)E(X)$, becomes $\text{Cov}(\epsilon, X) = E(E(\epsilon X \mid X)) - 0$, where we wrote the first term using iterated expectation and noted that the second term is zero given that $E(\epsilon \mid X) = 0$, which means the unconditional expectation of $\epsilon$ is also zero $E(\epsilon) = E(E(\epsilon \mid X)) = E(0) = 0$. By the same logic, the first term is also zero, which yields $\text{Cov}(\epsilon, X) = 0$.

The variance of $\hat{\beta}$ given $X$ is $\text{Var}(\hat{\beta} \mid X) = (X^TX)^{-1}X^T \text{Var}(\epsilon \mid X)X(X^TX)^{-1}$. $\text{Var}(\epsilon \mid X)$ is an $n$ x $n$ matrix that captures the variances (diagonal elements) and pairwise covariances (off-diagonal elements) of all elements of the error vector $\epsilon$, whose components are the individual **population-level** model errors $\epsilon_i$. $\text{Var}(\epsilon \mid X) = E[(\epsilon - E(\epsilon \mid X))(\epsilon - E(\epsilon \mid X))^T]$. 

If _homoscedasticity_ holds and errors are uncorrelated, all pairwise covariances (off-diagonal elements) are zero and all diagonal elements of $\text{Var}(\epsilon \mid X)$ have the same value $\sigma^2$. In that case, $\text{Var}(\hat{\beta \mid X}) = (X^TX)^{-1}X^T \sigma^2I_n X(X^TX)^{-1} = \sigma^2 (X^TX)^{-1}$. $\text{Var}(\hat{\beta \mid X}) is a $p$ x $p$ matrix whose diagonal entries are the variances of the corresponding regression coefficients. The off-diagonal elements are the pairwise covariances between regression coefficients.

Note that neither of these derivations requires that $\epsilon_i$ be normally distributed. What _is_ required is:
1. The model must be well specified ($E(\epsilon \mid X) = 0$), 
2. Errors must be uncorrelated ($\text{Cov}(\epsilon_i, \epsilon_j \mid X) = 0$), which is the same to say that the observations themselves must be _iid_, and 
3. Errors must be homoscedastic ($\text{Cov}(\epsilon_i, \epsilon_i \mid X) = \text{Var}(\epsilon_i \mid X) = 0$).

$\sigma^2$ is not observable, so we use $\hat{\sigma}^2 = \frac{1}{n-p}e^Te$, where $e^T = \langle e_1, e_2, ..., e_n \rangle = y - \hat{y} = y - Hy = (I_n - H)y$, with $I_n$ the $n$ x $n$ identity matrix and $H$ is the 'hat matrix' $H = X(X^TX)^{-1}X^T$.
