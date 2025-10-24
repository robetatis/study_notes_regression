# Hierarchical models

Based on [UCLA STATS 205](https://www.youtube.com/watch?v=flEIC4_bt8c&list=PLAYxx7zX5F1O2HbRr4gORnscbM9EszYbK&index=1)

## Starting point: the multiple linear model

The response is $y = \langle y_1, y_2, ..., y_n \rangle^T$, which is continuous.

We have a **fixed** design matrix $X$:

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

This setup is called the _fixed design problem_, where the modelling task is to find a function of $X$ that describes $E(y \mid X), where we assume that the distribution of $y$ along $X$ is always the same, which is the same as to say that $X$ is fixed. Otherwise we have the so-called _random design problem_.




