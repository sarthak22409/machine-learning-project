 function [J grad] = nnCostFunction(nn_params,input_layer_size, hidden_layer_size, num_labels, X, y, lambda)
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
  b = eye(10)               
  a1 = [ones(size(X,1),1) X];
  z1 = a1*Theta1';
  h1 = 1./(1+exp(-z1));
  a2 = [ones(size(h1,1),1) h1];
  z2= a2*Theta2';
  h2 = 1./(1+exp(-z2));
  m = size(X, 1);
  J=0;
  reg_term = 0;
  reg_term1 = 0;
  reg_term2 =0;
  y_Vec = (1:num_labels)==y;
  J = (1/m) * sum(sum((-y_Vec.*log(h2)-((1-y_Vec).*log(1-h2)))));
  
  for i = 2:size(Theta1,2)
    for j = 1:size(Theta1,1)
      reg_term1 = reg_term1 + Theta1(j,i).^2;
    endfor
  endfor
  for k = 2:size(Theta2,2)
    for l = 1:size(Theta2,1)
      reg_term2 = reg_term2 + Theta2(l,k).^2;
    endfor
  endfor
  reg_term = lambda/(2*m)*(reg_term1 + reg_term2);
  J = J+reg_term
  X = [ones(size(X,1),1) X];
  Theta1_grad =0
  Theta2_grad =0
  for t = 1:m
    a4 =  X(t,:)';
    z4 = Theta1 * a4;
    a5 = [1; sigmoid(z4)];
    z5 = Theta2 * a5;
    a6 = sigmoid(z5);
    yy = ([1:num_labels]==y(t))';
    delta_3 = a6 - yy;
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z4)];
    delta_2 = delta_2(2:end); % Taking of the bias row
    Theta1_grad = Theta1_grad + delta_2 * a4';
    Theta2_grad = Theta2_grad + delta_3 * a5';
    
  endfor
  
Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];


grad = [Theta1_grad(:) ; Theta2_grad(:)];
 
  
  
  
  
  
  
end
  
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



