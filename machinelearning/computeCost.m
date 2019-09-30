

function J = computeCost(X, y, theta)
  h = X*theta;
  m= length(y);
 
 
  squareerrors = (h-y).^2;
  J = 1/(2*m)*sum(squareerrors);
endfunction
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
 % number of training examples

% You need to return the following variables correctly 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.





% =========================================================================


