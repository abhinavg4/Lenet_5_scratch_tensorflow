function [input_img y_1 y_1_a y_1_maxpool y_2 y_2_a y_2_maxpool] = Forward(image,Weights_conv,activationFunction,Weights_bias)
%Forward pass through the lenet5 network
%image :- initial 28x28 image
%Weights_conv : Weights for initial conv layers
%activationFunction: Activtion function to be used
%Weights_bias : Biases of weights for conv

%input_img : image for fc to process
%y_1 : output of first conv layer
%y_1_a : output of first layer after applying activation function
%y_1_maxpool : output of first layer after maxpooling
%y_2,y_2_a,y_2_maxpool : all things similar as first layer for secind layer
    input_img = transpose(vec2mat(image,28));
    y_1 = vl_nnconv(single(input_img), Weights_conv{1}, Weights_bias{1}, 'pad', 2) ;
    y_1_a = activationFunction(y_1);
    y_1_maxpool = vl_nnpool(y_1_a,2,'stride',2);
    y_2 = vl_nnconv(y_1_maxpool,Weights_conv{2},Weights_bias{2});
    y_2_a = activationFunction(y_2);
    y_2_maxpool = vl_nnpool(y_2_a,2,'stride',2);
end