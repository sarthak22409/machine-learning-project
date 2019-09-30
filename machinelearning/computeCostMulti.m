function J = computeCostMulti(X, y, theta)
  a = mean(X(:,1));
  sigma = std(X(:,1);
  b =  (X(:,1) - a)./sigma;
  A = mean(X(:,2));
  sigma1 = std(X(:,2);
  B =  (X(:,2) - A)./sigma1;
  X_norm = [b B];
  m = length(y);
  h = X_norm*theta;
  
  squareerrors = (h-y).^2;
  J = 1/(2*m)*sum(squareerrors);
endfunction
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================

