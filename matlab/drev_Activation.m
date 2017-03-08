function y = drev_Activation(x)
% Derivative of the Activation Function.
%
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the derivative of the Activation function was
% applied element by element.
%a = 1 for ReLu and a=0 for Tanh
%%
    a=1;
    if a==0
        y = 1-Activation(x).^2;
    elseif a==2
       ac = Activation(x);
       y = (ac .* (1 - ac));
    else
        y = x > 0;
    end
end