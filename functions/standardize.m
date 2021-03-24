function y = standardize(x)

% Function to make your data have mean 0 and variance 1.
% Data in x are Txp, i.e. T time series observations times p variables

y = (x - repmat(mean(x,1),size(x,1),1))./repmat(std(x),size(x,1),1);