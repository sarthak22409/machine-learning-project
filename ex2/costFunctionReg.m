function [J, grad] = costFunctionReg(theta, X, y, lambda)
  h = X*theta;
  z = sigmoid(h)
  m = length(y);
  J = -1/(m)*sum(y.*log(z) + (1-y).*log(1-z)) +lambda/(2*m)*sum(theta(2:end).^2)
  
  a = length(theta);
  grad = zeros(size(theta));
  grad = 1/(m)*sum(X.*repmat((z-y),1,size(X,2)));
  grad(:,2:end) = grad(:,2:end) + lambda/(m)*(theta(:,2:end))';
  
  
  
  grad
  
endfunction
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================


