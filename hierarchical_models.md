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

This setup, where we condition on $X$ and thereofore assume it's **fixed (i.e., non-random)**, is called the _fixed design problem_, where the modelling task is to find a function of $X$ that describes $E(y \mid X)$. Setups where $X$ is not assumed fixed give rise to the so-called _random design problem_.

The model for $y$ is $y = X\beta + \epsilon$. Here, $y$ is $n$ x 1, $X$ is $n$ x $p$, $\beta$ is $p$ x 1 and $\epsilon$ is $n$ x 1.

$\epsilon$ is an $n$-dimensional **random error vector** $\epsilon = \langle \epsilon_1, ... \epsilon_n \rangle$, where each $\epsilon_i$ is a random variable and we assume they are _iid_ with $E(\epsilon_i) = 0$. We can further assume that $\epsilon_i \sim N(0, \sigma^2)$, i.e., that the $\epsilon_i$ all have the same distribution with mean zero and variance $\sigma^2$ (_homoscedasticity_).

With this setup, the parameter set defining the model is $\theta = (\beta ,\sigma^2)$. $\theta$ can be estimated either via $L2$ (ordinary
least squares OLS) or Maximum Likelihood Estimation MLE. OLS does not rely on any distributional assumptions for $\epsilon_i$, whereas MLE relies on $\epsilon_i \sim N(0, \sigma^2)$. Both estimation methods coincide under the previous assumption, i.e., $\hat{\beta}_{MLE} = \hat{\beta}_{OLS} = (X^TX)^{-1}X^Ty$. The result is a $p$ x 1 vector $\hat{\beta}^T = \langle \hat{\beta}_0, \hat{\beta}_1, ..., \hat{\beta}_{p-1} \rangle$
