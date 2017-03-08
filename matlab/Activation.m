function y = Activation(x)
% Activation function
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector with activation applied.
%
% a = 1 for ReLu and a=0 for Tanh
    a  = 1;
    if a ==1
       y = max(x,0);
    elseif a ==2
       y = 1 ./ (1 + exp(-x));
    else
       y = 1-(2./(1 + exp(2.*x)));
end