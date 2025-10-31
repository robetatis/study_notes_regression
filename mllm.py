# multi-level linear models (MLLMs)
# --------------------------------------------------------------------------

# intro
# -----------------------------------
# - used when data are naturally grouped (patients in hospitals, employees in companies/branches/departments, students in schools, etc.)
# - observations being grouped/nested means datapoints ARE NOT INDEPENDENT!!
# - in classical models, there's only one global slope and global intercepts for all observations
# - that means those parameters are *fixed*
# - this also implies that the relation between Y and X is the same across all observations, which with nested/grouped data is likely not true
# - another way to say this is that each value of Y is drawn from a single trend + random noise, i.e., we ignore groups
# - in a classical model, we could account for groups by using dummy variables and model their effect as fixed effects
# - however, this doesn't scale when we have many groups
# - in mllms we allow each group to have its own parameters (intercept and/or slope) and assume these params come from a common distribution
# - that is to say, in mllms, PARAMETERS HAVE DISTRIBUTIONS, AND ARE THEREFORE NOT FIXED NUMBERS
# - this way we acknowledge in the modelling that the data-generating process has structure across groups.
# - the distribution of the groups' parameters obviously have their own parameters -> hyperparameters.
# - for instance, group intercepts alpha_j ~ N(miu_alpha, sigma_alpha^2), here,  miu_alpha and sigma_alpha^2 are the mean and variance 
#   of the distribution from which we draw the group intercepts
# - there are 2 extremes when handling naturally nested data:
#   - pooled model - single model for all data -> biased estimates, ignores/wastes structure given by between-group differences
#   - unpooled model - one separate model per group -> needs lots of datapoints per group, small groups have noisy estimates
# - mllms are a middle ground: partial pooling -> each group has its own params BUT they all come from a single distribution
# - in partial pooling, groups with lots of data behave as having own separate regression, and groups with little data 'shrink' towards the global params
# - 'shrinkage':
#   - there's a global effect (global intercept, global slope) and each group's params are deviations (upwards or downwards) from the global params
#   - groups with little data see their effects 'shrink' towards zero, so these groups's effects are closer to the global trend
# - 

# model
# -----------------------------------
# two-level example, only one regressor x:
# y_ij = beta_0j + beta_1j*x_ij + epsilon_ij, epsilon_ij ~ N(0, sigma^2)
# beta_0j = gamma_00 + u_0j
# beta_1j = gamma_10 + u_1j