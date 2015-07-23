function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

hx = sigmoid(X * theta);
reg = (sum(theta'*theta)-theta(1)'*theta(1)) *lambda/(2*m);
J = sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m + reg;
%J = (sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m) + lambda * sum(theta(2:end).^2) / (2*m);
grad = (X' * (hx - y)/m) + (lambda/m) .* theta - (lambda/m) .* theta(1);
%grad =((hx - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;

end
