Plots showing the bias-variance tradeoff on the Franke function data with different data points and different methods.

The first set of figures illustrates the Bias Variance trade-off directly for OLS, Ridge and Lasso, their naming convention is given by:

someconvention

The second set of figures illustrates how the MSE changes when evaluated on training and test data, for different complexities, with high complexities corresponding to high variance and low bias, and low complexities corresponding to low variance and high bias as shown by the first set of figures. Their naming convention is given by:

methods_#datapoints_log10(lambda)_log10(noise)_compn(True/False)

The methods in the plot are written with initial letters only, e.g. OR -> OLS + Ridge, ORL -> OLS + Ridge + Lasso. compn is followed by False if compared to actual data, True if compared to noisy data
