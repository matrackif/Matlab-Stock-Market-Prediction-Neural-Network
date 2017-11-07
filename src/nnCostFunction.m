function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
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
X = [ones(m,1), X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Feed forward algorithm to get output of neural network h(x)
z2 = X * Theta1';
a2 = z2;
% Each row of a2 contains the node values of the second layer (a21, a22, a23...) corresponding
% to an input row from X 

% Add intercept term to a2 before we find a3

a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = z3;

J = zeros(m, 1);
for i = 1:size(y,1)
   rowy = y(i,:);
   rowa3 = a3(i, :);
   % rowy
   % rowa3
   J(i) = sum((rowa3 - rowy) .^ 2);
end

% Don't regularizethe terms that correspond to the bias. 
% For the matrices Theta1 and Theta2, this corresponds to the 1st column of each matrix
t1 = Theta1(:, 2:size(Theta1, 2));
t2 = Theta2(:, 2:size(Theta2, 2));
% size(t1)
% size(t2)
J = ((1 / (2 * m)) * sum(J)) + ((lambda / (2 * m)) * (sum(sum(t1.^2)) + sum(sum(t2.^2))));
%[val, p] = max(a3, [], 2);

% Compute gradients using back propagation
for i = 1:m
    rowa3 = a3(i, :);
    rowy = y(i, :);
    rowz2 = z2(i, :);
    delta3 = rowa3 - rowy;
   
    delta2 = Theta2' * delta3';
    % Remove delta2_0
    delta2 = delta2(2:end);
    delta2 = delta2 .* rowz2';
    
    
    rowa2 = a2(i, :);
    rowa1 = X(i, :);
    Theta2_grad = Theta2_grad + delta3' * rowa2;
    Theta1_grad = Theta1_grad + delta2 * rowa1;
end

Theta1_grad = Theta1_grad .* ( 1 / m);
Theta2_grad = Theta2_grad .* ( 1 / m);
% Don't regularize first column of gradient matrix
theta1_grad_reg = Theta1_grad(:, 2:end);
theta2_grad_reg = Theta2_grad(:, 2:end);

theta1_grad_reg = theta1_grad_reg + (lambda / m) .* t1;
theta2_grad_reg = theta2_grad_reg + (lambda / m) .* t2;

Theta1_grad(:, 2:end) = theta1_grad_reg;
Theta2_grad(:, 2:end) = theta2_grad_reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
